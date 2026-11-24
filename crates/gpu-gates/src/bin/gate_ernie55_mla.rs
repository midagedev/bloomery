//! GPU gate for ERNIE 5.5 Flash Preview's MLA-DSA latent attention, `bloomery_gpu_ernie55::attn_mla`
//! (its module doc's five steps) at the released file's shapes — 64 heads of 192 + 64, v 256,
//! kv_lora 512, q_lora 1024, hidden 4096, the indexer 64 heads of 128, top_k 2048, window 128,
//! layer 7 — over seeded inputs in a synthetic file (FP8_B128, BF16 and F32 tensors), nothing read
//! from a model file. The reference is the f64 rule of llama.cpp PR #31925's `build_mla_qk`,
//! `build_mla_out` and `build_mla_top_k` (`tools/ref/ernie55_ref.py`'s `_mla_attention`,
//! `indexer_scores`, `indexer_select`, `attend_sink`) over the file's dequantized weights and the
//! Q8_0 halves the load derives: every stage output rounded to f32 as the card holds it, the cache
//! rows and the flash's query rows rounded to f16 as the card stages them, a tie at the top-k's cut
//! to the lower row (the kernel's rule; numpy's `argpartition` orders none). [`BAND`] is
//! `gate_hy4_attn`'s: the card's f32 logit and value sums and `ex2.approx` under the f16 staging.
//!
//! 1. `repack`: the derived `attn_k_b`/`attn_v_b` planes read back are the host split's bytes bit
//!    for bit, and every repacked value is within its Q8_0 block's half step of the file's
//!    dequantized value (a cut along the wrong axis would permute them).
//! 2. `dense`: a 2,176-row state, one step at a time through the positions 0, 63 and 2,175 (counts
//!    1, 64 and 2,176, below the cut: no score launch, the list the identity over every key): the
//!    output within [`BAND`] of the rule, the cache rows the rule's f16 rows within [`F16_BAND`],
//!    no fault.
//! 3. `selected`: a 4,096-row state stepped through every position to 4,095: at 100 and 2,175
//!    (below the cut) the list is the identity and the window and the output the dense rule's;
//!    at 2,176 and 4,095 (counts 2,177 and 4,096) the list's top-k is the rule's set (ids exact)
//!    with the count `min(c, top_k) + window`, and the output within [`BAND`] of the rule over
//!    that list; the keys the rule's within [`KERNEL_BAND`].
//! 4. `prompt`: state A stepped 0..2,184, prompted in chunks of 48 and 37 to 4,080, stepped to
//!    4,096; state B prompted from 0 in chunks of 48, 37, 64, 8 and 1: B's outputs and cache rows
//!    are A's bits at every position, and A's stepped positions the selected state's; the same
//!    on a 2,176-row state, chunks only against steps only.
//! 5. `replay`: a step captured at position 0 and replayed at 100, 2,176 and 4,095, the position
//!    word rewritten, each replay the eager step's bits.
//! 6. `gate`: the output is the gated rule's within [`BAND`] and the ungated rule's off by more
//!    than the band at four positions.
//! 7. `refusals`: a mis-shaped site, a layer that is not a latent one, a span not the chunk's, no
//!    token, a position and rows past the state, the key kernel's turned span, a unit too wide,
//!    each refused by name before any launch.
//! 8. `faults`: a NaN input on a selecting state raises layer 7's `F32Product` (the latent key
//!    append) and `CacheValue` (the indexer key) and no finite output; a position past the
//!    cache raises `CachePos` from the key append and the index-key entry (its host twin's bit);
//!    a window wider than its block raises `KeyCount` and writes nothing.
//! 9. `shape`: the three new entries compile with no local depot.

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_ernie55_mla", gate::run())
}

#[cfg(feature = "gpu")]
mod gate {
    use bloomery_gpu::fused::FusedKernels;
    use bloomery_gpu::index_key::{IndexKeyKernels, IndexKeyLnNeoxArgs, index_key_ln_neox_host};
    use bloomery_gpu::indexer::IndexerF32Kernels;
    use bloomery_gpu::indexer::WindowArgs;
    use bloomery_gpu::rope_table::{Direction, RopeSpec, RopeTable};
    use bloomery_gpu::site::BY_NAME_COLS;
    use bloomery_gpu::weights::{DevWeight, Weights};
    use bloomery_gpu::{DeviceTensor, FaultSite, Gpu, GpuError, Window, WindowMut};
    use bloomery_gpu_ernie55::attn_mla::{
        Call, KvHalves, Mla, MlaState, NoVha, Shape, Sites, SpanWords, derive, split_kv_b,
    };
    use bloomery_gpu_gates::tier;
    use bloomery_gpu_gates::{
        GateError, GateRng, KERNEL_BAND, checks_failed, max_rel_err, no_local_depot, par_map,
        verdict, write_gguf,
    };
    use cuda_core::DeviceBuffer;
    use gguf::Split;
    use gguf::quant::{GgmlType, dequant_row, f32_to_f16_bits, fp32_to_bf16_bits, half_to_f32};
    use model::arch::delta_mla::{Mla as MlaDims, Situ};
    use model::arch::ernie55::hparams::{Hc, Hca, Hparams, Indexer, Rope};
    use model::arch::ernie55::names;
    use model::arch::names::{attn_k_b, attn_v_b};
    use model::arch::noaux::Experts;
    use std::mem::size_of;
    use std::path::Path;

    const NAME: &str = "gate_ernie55_mla";
    const HIDDEN: usize = 4096;
    const Q_LORA: usize = 1024;
    const KV_LORA: usize = 512;
    const NOPE: usize = 192;
    const ROPE: usize = 64;
    const HEAD_K: usize = NOPE + ROPE;
    const HEADS: usize = 64;
    const HEAD_V: usize = 256;
    const IDX_HEADS: usize = 64;
    const IDX_D: usize = 128;
    const TOP_K: usize = 2048;
    const WINDOW: usize = 128;
    const ROPE_BASE: f32 = 8e6;
    const EPS: f32 = 1e-6;
    /// The layer the file holds: the first MLA-DSA one.
    const LAYER: usize = 7;
    /// The selecting state's rows, and the dense state's (`top_k + window`: never selecting).
    const CTX: usize = 4096;
    const DENSE_CTX: usize = TOP_K + WINDOW;
    /// The scratch's columns a unit, and the largest prompt chunk.
    const UNIT: usize = 8;
    const MAX_T: usize = 64;
    const ROW: usize = ROPE + KV_LORA;
    const SCALE: f32 = 1.0 / 16.0;
    /// The indexer weights' scale, as the score pass forms it: `1/√(128 · 64)`.
    fn idx_scale() -> f32 {
        1.0 / ((IDX_D * IDX_HEADS) as f32).sqrt()
    }
    /// [derived] Against an f64 rule over the card's own f16 rows the error left is the f32 logit
    /// and value sums and `ex2.approx` (`gate_hy4_attn`'s derivation), the f32 stage outputs
    /// (≲ 1e-6 each) under them: 2e-4 at the largest output a launch writes.
    const BAND: f32 = 2e-4;
    /// [derived] The cache rows reach the card through f32 norms and ropes before their f16
    /// rounding: a value within an f32 rounding of an f16 boundary may flip one f16 ulp
    /// (2⁻¹¹ of its magnitude), so two ulps of the row's largest value.
    const F16_BAND: f32 = 1e-3;
    /// A repacked value's distance from its file value, in units of its block's `amax`: the Q8_0
    /// code's half step `amax / 254` and the f16 rounding of the block's scale (`amax · 2⁻¹¹`).
    const Q8_HALF_STEP: f64 = 1.0 / 254.0 + 1.0 / 1024.0;
    const SEED: u64 = 0x5eed_e55a;
    /// The weights' standard deviation: a projection's output is O(1) at these widths.
    const SIGMA: f32 = 1.0 / 64.0;
    const F: usize = size_of::<f32>();
    /// What a list the window pass refuses keeps.
    const LIST_POISON: u32 = u32::MAX;
    /// The positions the selected state is read at: two below the cut, two above it.
    const READ_AT: [usize; 4] = [100, DENSE_CTX - 1, DENSE_CTX, CTX - 1];

    // ------------------------------------------------------------- the file

    /// The 127 finite non-negative E4M3 values by code `0..=0x7e`, ascending.
    fn e4m3_magnitudes() -> Vec<f32> {
        (0u8..=0x7e)
            .map(|b| {
                let (exp, man) = ((b >> 3) & 0xf, b & 7);
                if exp == 0 {
                    f32::from(man) * 2f32.powi(-9)
                } else {
                    (1.0 + f32::from(man) / 8.0) * 2f32.powi(i32::from(exp) - 7)
                }
            })
            .collect()
    }

    /// The E4M3 code nearest `v` (the lower one at a tie); `table` is [`e4m3_magnitudes`].
    fn e4m3_code(v: f32, table: &[f32]) -> u8 {
        let mag = v.abs();
        let hi = table.partition_point(|&m| m <= mag);
        if hi == 0 {
            return 0;
        }
        let lo = hi - 1;
        let take_lo = hi >= table.len() || (mag - table[hi]).abs() >= (mag - table[lo]).abs();
        let code = if take_lo { lo } else { hi };
        code as u8 | if v.is_sign_negative() { 0x80 } else { 0 }
    }

    /// `rows × k` of N(0, SIGMA) in `ty`'s bytes: F32, BF16, or FP8_B128 (a block's scale
    /// `amax / 448`).
    fn tensor(seed: u64, rows: usize, k: usize, ty: GgmlType) -> Vec<u8> {
        let table = e4m3_magnitudes();
        par_map(rows, |r| {
            let mut rng = GateRng(seed ^ (r as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15));
            let x: Vec<f32> = (0..k.div_ceil(2))
                .flat_map(|_| {
                    let (u1, u2) = (rng.unit().max(1e-7), rng.unit());
                    let rad = (-2.0 * u1.ln()).sqrt() * SIGMA;
                    let ang = std::f32::consts::TAU * u2;
                    [rad * ang.cos(), rad * ang.sin()]
                })
                .take(k)
                .collect();
            let mut out = Vec::new();
            match ty {
                GgmlType::F32 => out.extend(x.iter().flat_map(|v| v.to_le_bytes())),
                GgmlType::BF16 => {
                    out.extend(x.iter().flat_map(|v| fp32_to_bf16_bits(*v).to_le_bytes()));
                }
                GgmlType::FP8_B128 => {
                    for blk in x.chunks(128) {
                        let amax = blk.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
                        let d = if amax > 0.0 { amax / 448.0 } else { 1.0 };
                        out.extend(d.to_le_bytes());
                        out.extend(blk.iter().map(|&v| e4m3_code(v / d, &table)));
                    }
                }
                _ => panic!("the gate writes F32, BF16 and FP8_B128"),
            }
            out
        })
        .into_iter()
        .flatten()
        .collect()
    }

    /// The E4M3 table against ggml's dequantizer, every code at scale 1.
    fn e4m3_table_agrees() -> Result<bool, GateError> {
        let mut bytes = Vec::new();
        bytes.extend(1.0f32.to_le_bytes());
        bytes.extend(0u8..=0x7e);
        bytes.push(0);
        let mut out = vec![0.0f32; 128];
        dequant_row(GgmlType::FP8_B128, &bytes, &mut out)?;
        let table = e4m3_magnitudes();
        let ok = (0u8..=0x7e).all(|b| out[usize::from(b)] == table[usize::from(b)]);
        println!(
            "{} e4m3 table: the gate's E4M3 magnitudes are ggml's for all 127 codes: {ok}",
            verdict(ok)
        );
        Ok(ok)
    }

    /// One tensor of the synthetic file.
    struct Decl {
        name: String,
        dims: Vec<u64>,
        ty: GgmlType,
        bytes: Vec<u8>,
    }

    fn decl(name: String, k: usize, rows: usize, ty: GgmlType, bytes: Vec<u8>) -> Decl {
        Decl {
            name,
            dims: vec![k as u64, rows as u64],
            ty,
            bytes,
        }
    }

    fn as_decl(d: &Decl) -> Decl {
        Decl {
            name: d.name.clone(),
            dims: d.dims.clone(),
            ty: d.ty,
            bytes: d.bytes.clone(),
        }
    }

    /// Layer `LAYER`'s tensors: the six dense projections and the fused `attn_kv_b` in FP8_B128
    /// (the checkpoint's type: the halves cut a head's 448 rows, three and a half scale blocks,
    /// at 192), the indexer key's and weights' projections in BF16, the vectors in F32.
    fn decls() -> Vec<Decl> {
        let fp8 = |name: String, k: usize, rows: usize, seed: u64| {
            decl(
                name,
                k,
                rows,
                GgmlType::FP8_B128,
                tensor(seed, rows, k, GgmlType::FP8_B128),
            )
        };
        let bf16 = |name: String, k: usize, rows: usize, seed: u64| {
            decl(
                name,
                k,
                rows,
                GgmlType::BF16,
                tensor(seed, rows, k, GgmlType::BF16),
            )
        };
        let f32 = |name: String, n: usize, seed: u64| {
            decl(name, n, 1, GgmlType::F32, tensor(seed, 1, n, GgmlType::F32))
        };
        let s = SEED;
        vec![
            fp8(names::attn_q_a(LAYER), HIDDEN, Q_LORA, s + 1),
            fp8(names::attn_q_b(LAYER), Q_LORA, HEADS * HEAD_K, s + 2),
            fp8(names::attn_kv_a_mqa(LAYER), HIDDEN, ROW, s + 3),
            fp8(
                names::attn_kv_b(LAYER),
                KV_LORA,
                HEADS * (NOPE + HEAD_V),
                s + 4,
            ),
            fp8(names::attn_gate(LAYER), Q_LORA, HEADS * HEAD_V, s + 5),
            fp8(names::attn_output(LAYER), HEADS * HEAD_V, HIDDEN, s + 6),
            bf16(
                names::indexer_attn_q_b(LAYER),
                Q_LORA,
                IDX_HEADS * IDX_D,
                s + 7,
            ),
            bf16(names::indexer_attn_k(LAYER), HIDDEN, IDX_D, s + 8),
            bf16(names::indexer_proj(LAYER), HIDDEN, IDX_HEADS, s + 9),
            f32(names::attn_q_a_norm(LAYER), Q_LORA, s + 10),
            f32(names::attn_kv_a_norm(LAYER), KV_LORA, s + 11),
            f32(names::attn_sinks(LAYER), HEADS, s + 12),
            f32(names::indexer_k_norm(LAYER), IDX_D, s + 13),
            f32(names::indexer_k_norm_bias(LAYER), IDX_D, s + 14),
        ]
    }

    /// The hyperparameters of the synthetic file: ERNIE's widths, one latent layer.
    fn hparams() -> Hparams {
        Hparams {
            n_layer: LAYER + 1,
            n_embd: HIDDEN,
            n_vocab: 1,
            n_ctx_train: CTX,
            dense_lead: 1,
            dense_ff: 0,
            rms_eps: EPS,
            heads: HEADS,
            mla: MlaDims {
                q_lora: Q_LORA,
                kv_lora: KV_LORA,
                head_k: HEAD_K,
                head_v: HEAD_V,
                pe: ROPE,
            },
            hca: Hca {
                head: 512,
                ratio: 128,
                window: WINDOW,
                o_groups: 8,
                o_lora: 256,
            },
            ratios: (0..=LAYER)
                .map(|l| if l == LAYER { 0 } else { 128 })
                .collect(),
            indexer: Indexer {
                heads: IDX_HEADS,
                d: IDX_D,
                top_k: TOP_K,
            },
            vha_rank: 64,
            rope: Rope {
                base: ROPE_BASE,
                compress_base: 1e7,
                factor: 1.0,
                original_ctx: CTX,
                beta_fast: 32.0,
                beta_slow: 1.0,
            },
            experts: Experts {
                n_expert: 512,
                n_used: 10,
                ff: 1,
                n_shared: 1,
                shared_width: 1,
                weights_scale: 1.0,
                weights_norm: true,
            },
            latent: 2048,
            situ: Situ {
                gate_beta: 1.0,
                value_beta: 1.0,
            },
            hc: Hc {
                streams: 4,
                eps: 1e-6,
                sinkhorn_iters: 4,
            },
            defaults: Vec::new(),
        }
    }

    // ------------------------------------------------------------- the rule

    /// The file's weights dequantized (f32), and the derived halves from the split's Q8_0 bytes.
    struct WeightsF {
        q_a: Vec<f32>,
        q_b: Vec<f32>,
        kv_a: Vec<f32>,
        gate: Vec<f32>,
        o: Vec<f32>,
        idx_q: Vec<f32>,
        idx_k: Vec<f32>,
        idx_w: Vec<f32>,
        q_a_norm: Vec<f32>,
        kv_a_norm: Vec<f32>,
        sinks: Vec<f32>,
        kn_w: Vec<f32>,
        kn_b: Vec<f32>,
        /// `[HEADS][NOPE][KV_LORA]`.
        k_b: Vec<f32>,
        /// `[HEADS][HEAD_V][KV_LORA]`.
        v_b: Vec<f32>,
    }

    /// A decl's rows dequantized.
    fn rows_of(d: &Decl) -> Result<Vec<f32>, GateError> {
        let k = d.dims[0] as usize;
        let rows = d.dims[1] as usize;
        let row_bytes = d.bytes.len() / rows;
        let mut out = vec![0.0f32; k * rows];
        for (r, o) in out.chunks_mut(k).enumerate() {
            dequant_row(d.ty, &d.bytes[r * row_bytes..(r + 1) * row_bytes], o)?;
        }
        Ok(out)
    }

    fn find<'a>(decls: &'a [Decl], name: &str) -> Result<&'a Decl, GateError> {
        decls
            .iter()
            .find(|d| d.name == name)
            .ok_or_else(|| format!("the synthetic file lacks {name}").into())
    }

    /// The Q8_0 bytes of one derived half dequantized.
    fn q8_rows(bytes: &[u8], k: usize) -> Result<Vec<f32>, GateError> {
        let row_bytes = k / 32 * 34;
        let rows = bytes.len() / row_bytes;
        let mut out = vec![0.0f32; k * rows];
        for (r, o) in out.chunks_mut(k).enumerate() {
            dequant_row(
                GgmlType::Q8_0,
                &bytes[r * row_bytes..(r + 1) * row_bytes],
                o,
            )?;
        }
        Ok(out)
    }

    /// `W · x` in f64 (`W` `rows × k`), rounded to f32.
    fn matvec(w: &[f32], x: &[f32], k: usize) -> Vec<f32> {
        par_map(w.len() / k, |r| {
            w[r * k..(r + 1) * k]
                .iter()
                .zip(x)
                .map(|(&a, &b)| f64::from(a) * f64::from(b))
                .sum::<f64>() as f32
        })
    }

    fn rms_norm(x: &[f32], gain: &[f32]) -> Vec<f32> {
        let ms = x.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / x.len() as f64;
        let inv = (1.0 / (ms + f64::from(EPS)).sqrt()) as f32;
        x.iter().zip(gain).map(|(&v, &g)| v * inv * g).collect()
    }

    fn layer_norm(x: &[f32], w: &[f32], b: &[f32]) -> Vec<f32> {
        let mu = x.iter().map(|&v| f64::from(v)).sum::<f64>() / x.len() as f64;
        let var = x.iter().map(|&v| (f64::from(v) - mu).powi(2)).sum::<f64>() / x.len() as f64;
        let inv = 1.0 / (var + f64::from(EPS)).sqrt();
        x.iter()
            .zip(w)
            .zip(b)
            .map(|((&v, &g), &bi)| (((f64::from(v) - mu) * inv) as f32) * g + bi)
            .collect()
    }

    /// The interleaved turn of `x`'s first `n` values by the table row `cs`.
    fn rope_norm(x: &mut [f32], cs: &[f32], n: usize) {
        for i in 0..n / 2 {
            let (c, s) = (f64::from(cs[2 * i]), f64::from(cs[2 * i + 1]));
            let (x0, x1) = (f64::from(x[2 * i]), f64::from(x[2 * i + 1]));
            x[2 * i] = (x0 * c - x1 * s) as f32;
            x[2 * i + 1] = (x0 * s + x1 * c) as f32;
        }
    }

    /// The NEOX turn of `x`'s first `n` values (pairs `(i, i + n/2)`) by the table row `cs`.
    fn rope_neox(x: &mut [f32], cs: &[f32], n: usize) {
        for i in 0..n / 2 {
            let (c, s) = (f64::from(cs[2 * i]), f64::from(cs[2 * i + 1]));
            let (x0, x1) = (f64::from(x[i]), f64::from(x[i + n / 2]));
            x[i] = (x0 * c - x1 * s) as f32;
            x[i + n / 2] = (x0 * s + x1 * c) as f32;
        }
    }

    fn rope_row(rope: &RopeTable, p: u32) -> Vec<f32> {
        let mut cs = Vec::with_capacity(ROPE);
        rope.push(p, Direction::Forward, &mut cs);
        cs
    }

    /// The f64 rule over the weights, with its own key caches for every position.
    struct Rule {
        w: WeightsF,
        rope: RopeTable,
        /// `[CTX][ROW]` f16 bits.
        kv: Vec<u16>,
        /// `[CTX][IDX_D]`.
        ik: Vec<f32>,
    }

    /// A position's query side: the gate, the roped query, the absorbed query, the turned
    /// indexer query and the scaled weights.
    struct Query {
        gate: Vec<f32>,
        q: Vec<f32>,
        q_abs: Vec<f32>,
        iq: Vec<f32>,
        iw: Vec<f32>,
    }

    impl Rule {
        fn new(w: WeightsF, xs: &[f32]) -> Result<Rule, GateError> {
            let rope = RopeTable::new(&RopeSpec::window(ROPE_BASE, ROPE))?;
            let rows = par_map(CTX, |p| {
                let x = &xs[p * HIDDEN..(p + 1) * HIDDEN];
                let cs = rope_row(&rope, p as u32);
                let kvpe = matvec(&w.kv_a, x, HIDDEN);
                let kvc = rms_norm(&kvpe[..KV_LORA], &w.kv_a_norm);
                let mut k_pe = kvpe[KV_LORA..].to_vec();
                rope_norm(&mut k_pe, &cs, ROPE);
                let row: Vec<u16> = k_pe
                    .iter()
                    .chain(&kvc)
                    .map(|&v| f32_to_f16_bits(v))
                    .collect();
                let mut ik = layer_norm(&matvec(&w.idx_k, x, HIDDEN), &w.kn_w, &w.kn_b);
                rope_neox(&mut ik, &cs, ROPE);
                (row, ik)
            });
            Ok(Rule {
                w,
                rope,
                kv: rows.iter().flat_map(|(r, _)| r.iter().copied()).collect(),
                ik: rows.iter().flat_map(|(_, k)| k.iter().copied()).collect(),
            })
        }

        fn query(&self, p: usize, x: &[f32]) -> Query {
            let w = &self.w;
            let qr = rms_norm(&matvec(&w.q_a, x, HIDDEN), &w.q_a_norm);
            let gate = matvec(&w.gate, &qr, Q_LORA)
                .into_iter()
                .map(|v| (1.0 / (1.0 + (-f64::from(v)).exp())) as f32)
                .collect();
            let cs = rope_row(&self.rope, p as u32);
            let mut q = matvec(&w.q_b, &qr, Q_LORA);
            for h in 0..HEADS {
                rope_norm(&mut q[h * HEAD_K + NOPE..(h + 1) * HEAD_K], &cs, ROPE);
            }
            let q_abs: Vec<f32> = par_map(HEADS * KV_LORA, |i| {
                let (h, j) = (i / KV_LORA, i % KV_LORA);
                (0..NOPE)
                    .map(|d| {
                        f64::from(w.k_b[(h * NOPE + d) * KV_LORA + j])
                            * f64::from(q[h * HEAD_K + d])
                    })
                    .sum::<f64>() as f32
            });
            let mut iq = matvec(&w.idx_q, &qr, Q_LORA);
            for h in 0..IDX_HEADS {
                rope_neox(&mut iq[h * IDX_D..(h + 1) * IDX_D], &cs, ROPE);
            }
            let iw = matvec(&w.idx_w, x, HIDDEN)
                .into_iter()
                .map(|v| v * idx_scale())
                .collect();
            Query {
                gate,
                q,
                q_abs,
                iq,
                iw,
            }
        }

        /// The candidates a query at `p` scores: the positions older than its window.
        fn cands(p: usize) -> usize {
            (p + 1).saturating_sub(WINDOW)
        }

        /// The indexer's scores of `0 .. cands` at `p`.
        fn scores(&self, q: &Query, p: usize) -> Vec<f64> {
            par_map(Self::cands(p), |s| {
                let k = &self.ik[s * IDX_D..(s + 1) * IDX_D];
                (0..IDX_HEADS)
                    .map(|h| {
                        let d: f64 = (0..IDX_D)
                            .map(|i| f64::from(q.iq[h * IDX_D + i]) * f64::from(k[i]))
                            .sum();
                        d.max(0.0) * f64::from(q.iw[h])
                    })
                    .sum()
            })
        }

        /// The rule's top-k of `scores` (a tie to the lower row), ascending.
        fn top_k(scores: &[f64], k: usize) -> Vec<usize> {
            let mut idx: Vec<usize> = (0..scores.len()).collect();
            idx.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
            idx.truncate(k);
            idx.sort_unstable();
            idx
        }

        /// The keys a query at `p` attends on a selecting state: `None` below the cut (every
        /// key, the dense form); above it the top-k of the candidates and the window, in the
        /// device list's order. The second is the rule's score gap at the cut.
        fn select(&self, q: &Query, p: usize) -> (Option<Vec<usize>>, Option<f64>) {
            let c = Self::cands(p);
            if c <= TOP_K {
                return (None, None);
            }
            let sc = self.scores(q, p);
            // The gap reads a sorted copy: the top-k takes the scores by candidate row.
            let mut sorted = sc.clone();
            sorted.sort_by(|a, b| b.total_cmp(a));
            let gap = sorted[TOP_K - 1] - sorted[TOP_K];
            let mut keys = Self::top_k(&sc, TOP_K);
            keys.extend(c..=p);
            (Some(keys), Some(gap))
        }

        fn kv_at(&self, r: usize, d: usize) -> f64 {
            f64::from(half_to_f32(self.kv[r * ROW + d]))
        }

        /// The heads' latent attention of a query over `keys` (`[HEADS][KV_LORA]`, f32).
        fn attend(&self, q: &Query, keys: &[usize]) -> Vec<f32> {
            par_map(HEADS, |h| {
                let qc: Vec<f64> = (0..ROW)
                    .map(|d| {
                        let v = if d < ROPE {
                            q.q[h * HEAD_K + NOPE + d]
                        } else {
                            q.q_abs[h * KV_LORA + d - ROPE]
                        };
                        f64::from(half_to_f32(f32_to_f16_bits(v)))
                    })
                    .collect();
                let logits: Vec<f64> = keys
                    .iter()
                    .map(|&r| {
                        (0..ROW).map(|d| qc[d] * self.kv_at(r, d)).sum::<f64>() * f64::from(SCALE)
                    })
                    .collect();
                let sink = f64::from(self.w.sinks[h]);
                let m = logits.iter().copied().fold(sink, f64::max);
                let ws: Vec<f64> = logits.iter().map(|&l| (l - m).exp()).collect();
                let den = ws.iter().sum::<f64>() + (sink - m).exp();
                (0..KV_LORA)
                    .map(|d| {
                        keys.iter()
                            .zip(&ws)
                            .map(|(&r, &wj)| wj * self.kv_at(r, ROPE + d))
                            .sum::<f64>()
                            / den
                    })
                    .collect::<Vec<f64>>()
            })
            .into_iter()
            .flatten()
            .map(|v| v as f32)
            .collect()
        }

        /// The heads' values of `att` through `v_b` (`[HEADS][HEAD_V]`, f32).
        fn values(&self, att: &[f32]) -> Vec<f32> {
            par_map(HEADS * HEAD_V, |i| {
                let (h, v) = (i / HEAD_V, i % HEAD_V);
                (0..KV_LORA)
                    .map(|j| {
                        f64::from(self.w.v_b[(h * HEAD_V + v) * KV_LORA + j])
                            * f64::from(att[h * KV_LORA + j])
                    })
                    .sum::<f64>() as f32
            })
        }

        /// The layer's output at `p` over `keys`: the attention, the values, the head mix (the
        /// identity here), the output gate, `o_proj`.
        fn output(&self, q: &Query, keys: &[usize]) -> Vec<f32> {
            let o = self.values(&self.attend(q, keys));
            let gated: Vec<f32> = o.iter().zip(&q.gate).map(|(&a, &g)| a * g).collect();
            matvec(&self.w.o, &gated, HEADS * HEAD_V)
        }

        /// The output with the gate left out.
        fn output_ungated(&self, q: &Query, keys: &[usize]) -> Vec<f32> {
            let o = self.values(&self.attend(q, keys));
            matvec(&self.w.o, &o, HEADS * HEAD_V)
        }
    }

    // ------------------------------------------------------------ the device

    fn clause(name: &str, pass: bool, detail: &str) -> bool {
        println!("{} {name}: {detail}", verdict(pass));
        pass
    }

    fn banded(name: &str, y: &[f32], want: &[f32], band: f32) -> bool {
        match max_rel_err(y, want) {
            Ok(rel) => clause(
                name,
                rel <= band,
                &format!("max_rel {rel:.3e} (band {band:.0e})"),
            ),
            Err(e) => clause(name, false, &e.to_string()),
        }
    }

    /// The f16 cache rows as f32.
    fn f16_rows(bits: &[u16]) -> Vec<f32> {
        bits.iter().map(|&b| half_to_f32(b)).collect()
    }

    /// The card, the component, its forms, buffers and words, the inputs on the card, and the
    /// one-token buffers a captured step reads and writes.
    struct Dev {
        gpu: Gpu,
        mla: Mla,
        bn: bloomery_gpu::site::ByName,
        s: bloomery_gpu_ernie55::attn_mla::Scratch,
        words: SpanWords,
        vha: NoVha,
        xs: DeviceBuffer<f32>,
        x1: DeviceBuffer<f32>,
        o1: DeviceBuffer<f32>,
        oc: DeviceBuffer<f32>,
        x_cap: DeviceBuffer<f32>,
        o_cap: DeviceBuffer<f32>,
    }

    impl Dev {
        /// One step at `pos` over the input row `x` (`HIDDEN` values): its output.
        fn step(
            &mut self,
            w: &Weights,
            st: &mut MlaState,
            pos: u32,
            x: &[f32],
        ) -> Result<Vec<f32>, GateError> {
            let Dev {
                gpu,
                mla,
                bn,
                s,
                words,
                vha,
                x1,
                o1,
                ..
            } = self;
            let stream = gpu.stream();
            x1.copy_from_host(stream, x)?;
            words.set(stream, pos, 1)?;
            mla.enqueue_step(
                &mut Call {
                    gpu,
                    w,
                    vha,
                    words,
                    bn,
                    s,
                },
                st,
                pos,
                x1,
                o1,
            )?;
            stream.synchronize()?;
            Ok(o1.to_host_vec(stream)?)
        }

        /// A prompt chunk of `t` tokens at `pos0` over the inputs on the card: its outputs.
        fn prompt(
            &mut self,
            w: &Weights,
            st: &mut MlaState,
            pos0: u32,
            t: usize,
        ) -> Result<Vec<f32>, GateError> {
            let Dev {
                gpu,
                mla,
                bn,
                s,
                words,
                vha,
                xs,
                oc,
                ..
            } = self;
            let stream = gpu.stream();
            let x = Window::<f32>::of(xs, pos0 as usize * HIDDEN * F, t * HIDDEN)?;
            words.set(stream, pos0, t)?;
            {
                let mut o = WindowMut::<f32>::of_mut(oc, 0, t * HIDDEN)?;
                mla.enqueue_prompt(
                    &mut Call {
                        gpu,
                        w,
                        vha,
                        words,
                        bn,
                        s,
                    },
                    st,
                    pos0,
                    t,
                    &x,
                    &mut o,
                )?;
            }
            stream.synchronize()?;
            Ok(Window::<f32>::of(oc, 0, t * HIDDEN)?.to_host_vec(stream)?)
        }

        /// The output buffer of a captured step.
        fn read_cap(&self) -> Result<Vec<f32>, GateError> {
            self.gpu.stream().synchronize()?;
            Ok(self.o_cap.to_host_vec(self.gpu.stream())?)
        }

        /// The state's cache rows.
        fn cache_rows(&self, st: &MlaState) -> Result<Vec<u16>, GateError> {
            self.gpu.stream().synchronize()?;
            Ok(st.cache().buf().to_host_vec(self.gpu.stream())?)
        }

        /// The state's index keys.
        fn key_rows(&self, st: &MlaState) -> Result<Vec<f32>, GateError> {
            self.gpu.stream().synchronize()?;
            Ok(st.keys().buf().to_host_vec(self.gpu.stream())?)
        }

        /// No fault raised.
        fn clean(&self, name: &str) -> Result<bool, GateError> {
            let f = self.gpu.fault()?;
            Ok(clause(
                &format!("{name} no_fault"),
                f.is_none(),
                &format!("{f:?}"),
            ))
        }
    }

    /// The seeded inputs, `[CTX][HIDDEN]`, each row a position's normed layer input.
    fn inputs() -> Vec<f32> {
        par_map(CTX, |p| {
            let mut rng = GateRng(SEED ^ (p as u64).wrapping_mul(0xd1b5_4a32_d192_ed03));
            (0..HIDDEN)
                .map(|_| rng.uni(-1.0, 1.0))
                .collect::<Vec<f32>>()
        })
        .into_iter()
        .flatten()
        .collect()
    }

    /// The chunks a prompt is cut into from `from` to `end`: `sizes` cycling, the last trimmed.
    fn chunks(sizes: &[usize], from: usize, end: usize) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        let mut at = from;
        let mut next = sizes.iter().cycle();
        while at < end {
            let size = (*next.next().expect("sizes holds a chunk")).min(end - at);
            out.push((at, size));
            at += size;
        }
        out
    }

    /// A run over `xs` to `end` rows: the step form through `to_step`, chunks of `sizes` to
    /// `chunk_end`, steps to `end`: every position's output (`[end][HIDDEN]`).
    fn drive(
        dev: &mut Dev,
        w: &Weights,
        st: &mut MlaState,
        xs: &[f32],
        (to_step, chunk_end, end): (usize, usize, usize),
        sizes: &[usize],
    ) -> Result<Vec<f32>, GateError> {
        let mut outs = vec![0.0f32; end * HIDDEN];
        for p in 0..to_step {
            let o = dev.step(w, st, p as u32, &xs[p * HIDDEN..(p + 1) * HIDDEN])?;
            outs[p * HIDDEN..(p + 1) * HIDDEN].copy_from_slice(&o);
        }
        for (at, n) in chunks(sizes, to_step, chunk_end) {
            let o = dev.prompt(w, st, at as u32, n)?;
            outs[at * HIDDEN..(at + n) * HIDDEN].copy_from_slice(&o);
        }
        for p in chunk_end..end {
            let o = dev.step(w, st, p as u32, &xs[p * HIDDEN..(p + 1) * HIDDEN])?;
            outs[p * HIDDEN..(p + 1) * HIDDEN].copy_from_slice(&o);
        }
        Ok(outs)
    }

    /// The dense keys of a query at `p`.
    fn all_keys(p: usize) -> Vec<usize> {
        (0..=p).collect()
    }

    fn row(outs: &[f32], p: usize) -> &[f32] {
        &outs[p * HIDDEN..(p + 1) * HIDDEN]
    }

    // --------------------------------------------------------------- clauses

    /// Clause 1: the derived halves: the resident planes the host split's bytes, every value within its
    /// block's half step of the file's.
    fn repack(gpu: &Gpu, w: &Weights, halves: &KvHalves, kv_b: &[f32]) -> Result<bool, GateError> {
        let stream = gpu.stream();
        stream.synchronize()?;
        let mut ok = true;
        for (name, (info, bytes)) in [attn_k_b(LAYER), attn_v_b(LAYER)].iter().zip(&halves.parts) {
            let DevWeight::Q8_0 { qs, d, k } = w.resident(NAME, name)? else {
                return Err(format!("{name} is not resident as Q8_0").into());
            };
            let words = qs.buf().to_host_vec(stream)?;
            let scales = d.buf().to_host_vec(stream)?;
            let (rows, blocks) = (d.rows(), k / 32);
            let mut back = Vec::with_capacity(rows * blocks * 34);
            for r in 0..rows {
                for b in 0..blocks {
                    back.extend_from_slice(&scales[r * blocks + b].to_le_bytes());
                    for i in 0..32 {
                        let j = b * 32 + i;
                        back.push((words[r * (k / 4) + j / 4] >> (8 * (j % 4))) as u8);
                    }
                }
            }
            ok &= clause(
                &format!("repack {name} planes"),
                back == *bytes && info.dims == [*k as u64, rows as u64] && *name == info.name,
                &format!(
                    "{} bytes read back, {} on the host; dims {:?}",
                    back.len(),
                    bytes.len(),
                    info.dims
                ),
            );
        }
        let k_rows = q8_rows(&halves.parts[0].1, NOPE)?;
        let v_rows = q8_rows(&halves.parts[1].1, KV_LORA)?;
        let per_head = NOPE + HEAD_V;
        let worst = par_map(HEADS, |h| {
            let mut worst = 0.0f64;
            // k_b: block (j, b) of row `h · kv_lora + j` is the file's head rows `32b ..`, column j.
            for j in 0..KV_LORA {
                for b in 0..NOPE / 32 {
                    let file: Vec<f64> = (0..32)
                        .map(|l| f64::from(kv_b[(h * per_head + 32 * b + l) * KV_LORA + j]))
                        .collect();
                    let amax = file.iter().fold(0.0f64, |a, &v| a.max(v.abs()));
                    for (l, v) in file.iter().enumerate() {
                        let q = f64::from(k_rows[(h * KV_LORA + j) * NOPE + 32 * b + l]);
                        worst = worst.max((q - v).abs() / (amax * Q8_HALF_STEP + 1e-7));
                    }
                }
            }
            // v_b: row `h · v + v` is the file's head row `nope + v`, block by block.
            for v in 0..HEAD_V {
                for blk in 0..KV_LORA / 32 {
                    let file = &kv_b[(h * per_head + NOPE + v) * KV_LORA + blk * 32..][..32];
                    let amax = file.iter().fold(0.0f64, |a, &x| a.max(f64::from(x).abs()));
                    for i in 0..32 {
                        let q = f64::from(v_rows[(h * HEAD_V + v) * KV_LORA + blk * 32 + i]);
                        worst = worst
                            .max((q - f64::from(file[i])).abs() / (amax * Q8_HALF_STEP + 1e-7));
                    }
                }
            }
            worst
        })
        .into_iter()
        .fold(0.0f64, f64::max);
        ok &= clause(
            "repack half step",
            worst <= 1.0,
            &format!("the worst repacked value {worst:.3} half steps from its file value"),
        );
        Ok(ok)
    }

    /// Clause 2: the dense form: a state below the cut, stepped through every position; the rule's
    /// outputs at 0, 63 and 2,175 and its cache rows everywhere.
    fn dense(
        dev: &mut Dev,
        w: &Weights,
        st: &mut MlaState,
        xs: &[f32],
        rule: &Rule,
    ) -> Result<bool, GateError> {
        let mut ok = true;
        for p in 0..DENSE_CTX {
            let o = dev.step(w, st, p as u32, row(xs, p))?;
            if [0, 63, DENSE_CTX - 1].contains(&p) {
                let q = rule.query(p, row(xs, p));
                let want = rule.output(&q, &all_keys(p));
                ok &= banded(
                    &format!("dense p={p} out (count {})", p + 1),
                    &o,
                    &want,
                    BAND,
                );
            }
        }
        let cache = dev.cache_rows(st)?;
        ok &= banded(
            "dense cache rows",
            &f16_rows(&cache),
            &f16_rows(&rule.kv[..DENSE_CTX * ROW]),
            F16_BAND,
        );
        ok &= dev.clean("dense")?;
        Ok(ok)
    }

    /// Clause 3: the selected form: a state above the cut, stepped through every position; the list,
    /// its count and the output at the read positions; the keys everywhere. Returns the run's
    /// outputs for the clauses that follow.
    fn selected(
        dev: &mut Dev,
        w: &Weights,
        st: &mut MlaState,
        xs: &[f32],
        rule: &Rule,
    ) -> Result<(bool, Vec<f32>), GateError> {
        let mut ok = true;
        let mut outs = vec![0.0f32; CTX * HIDDEN];
        for p in 0..CTX {
            let o = dev.step(w, st, p as u32, row(xs, p))?;
            if READ_AT.contains(&p) {
                let q = rule.query(p, row(xs, p));
                let (sel, gap) = rule.select(&q, p);
                let c = Rule::cands(p);
                let (keys, want_count, cut) = match &sel {
                    None => (all_keys(p), p + 1, "below the cut".to_string()),
                    Some(keys) => (
                        keys.clone(),
                        TOP_K + WINDOW,
                        format!(
                            "top-k {TOP_K} of c={c}, the rule's gap at the cut {}",
                            gap.map_or("none".to_string(), |g| format!("{g:.3e}"))
                        ),
                    ),
                };
                let (list, count) = dev.s.selection(dev.gpu.stream(), 0)?;
                let listed: Vec<usize> =
                    list[..count as usize].iter().map(|&v| v as usize).collect();
                let first_off = listed
                    .iter()
                    .zip(&keys)
                    .position(|(a, b)| a != b)
                    .map_or("none".to_string(), |i| i.to_string());
                ok &= clause(
                    &format!("selected p={p} list ({cut})"),
                    listed == keys && count as usize == want_count,
                    &format!(
                        "{} listed of count {count}, the rule's {}; first differing id at {first_off}",
                        listed.len(),
                        keys.len()
                    ),
                );
                let want = rule.output(&q, &keys);
                ok &= banded(&format!("selected p={p} out"), &o, &want, BAND);
            }
            outs[p * HIDDEN..(p + 1) * HIDDEN].copy_from_slice(&o);
        }
        let keys = dev.key_rows(st)?;
        ok &= banded("selected keys", &keys, &rule.ik, KERNEL_BAND);
        ok &= dev.clean("selected")?;
        Ok((ok, outs))
    }

    /// Clause 4: the prompt form: a run of steps and chunks against a run of chunks only, and its
    /// stepped positions against the selected state's outputs.
    fn prompt(dev: &mut Dev, w: &Weights, xs: &[f32], outs_s: &[f32]) -> Result<bool, GateError> {
        let to_step = DENSE_CTX + 8;
        let chunk_end = CTX - 16;
        let mut a = dev.mla.state(dev.gpu.stream(), 0, CTX)?;
        let outs_a = drive(dev, w, &mut a, xs, (to_step, chunk_end, CTX), &[48, 37])?;
        let cache_a = dev.cache_rows(&a)?;
        let mut b = dev.mla.state(dev.gpu.stream(), 0, CTX)?;
        let outs_b = drive(dev, w, &mut b, xs, (0, CTX, CTX), &[48, 37, 64, 8, 1])?;
        let cache_b = dev.cache_rows(&b)?;
        let mut ok = true;
        let first = (0..CTX).find(|&p| row(&outs_a, p) != row(&outs_b, p));
        ok &= clause(
            "prompt outputs",
            first.is_none(),
            &format!(
                "chunks only against steps and chunks over {CTX} positions, first differing \
                 position {first:?}"
            ),
        );
        ok &= clause(
            "prompt cache rows",
            cache_a == cache_b,
            &format!(
                "{} f16 rows, {} equal",
                cache_a.len(),
                cache_a.iter().zip(&cache_b).filter(|(a, b)| a == b).count()
            ),
        );
        let stepped = (0..to_step)
            .chain(chunk_end..CTX)
            .all(|p| row(&outs_a, p) == row(outs_s, p));
        ok &= clause(
            "prompt stepped positions",
            stepped,
            "A's stepped positions are the selected state's bits",
        );
        // A state below the cut takes the same list form: chunks only against steps only.
        let mut da = dev.mla.state(dev.gpu.stream(), 0, DENSE_CTX)?;
        let outs_da = drive(dev, w, &mut da, xs, (DENSE_CTX, DENSE_CTX, DENSE_CTX), &[1])?;
        let cache_da = dev.cache_rows(&da)?;
        let mut db = dev.mla.state(dev.gpu.stream(), 0, DENSE_CTX)?;
        let outs_db = drive(
            dev,
            w,
            &mut db,
            xs,
            (0, DENSE_CTX, DENSE_CTX),
            &[48, 37, 64, 8, 1],
        )?;
        let cache_db = dev.cache_rows(&db)?;
        let first = (0..DENSE_CTX).find(|&p| row(&outs_da, p) != row(&outs_db, p));
        ok &= clause(
            "prompt dense outputs",
            first.is_none() && cache_da == cache_db,
            &format!(
                "a {DENSE_CTX}-row state, chunks only against steps only: first differing \
                 position {first:?}, cache rows equal {}",
                cache_da == cache_db
            ),
        );
        ok &= dev.clean("prompt")?;
        Ok(ok)
    }

    /// Clause 5: a captured step replayed at other positions.
    fn replay(dev: &mut Dev, w: &Weights, xs: &[f32], outs_s: &[f32]) -> Result<bool, GateError> {
        let mut c = dev.mla.state(dev.gpu.stream(), 0, CTX)?;
        dev.words.set(dev.gpu.stream(), 0, 1)?;
        let graph = {
            let Dev {
                gpu,
                mla,
                bn,
                s,
                words,
                vha,
                x_cap,
                o_cap,
                ..
            } = dev;
            gpu.capture(|_| {
                mla.enqueue_step(
                    &mut Call {
                        gpu,
                        w,
                        vha,
                        words,
                        bn,
                        s,
                    },
                    &mut c,
                    0,
                    x_cap,
                    o_cap,
                )
            })?
        };
        let mut ok = true;
        let mut done = 0usize;
        for p in [100usize, DENSE_CTX, CTX - 1] {
            for r in done..p {
                dev.step(w, &mut c, r as u32, row(xs, r))?;
            }
            done = p;
            dev.x_cap.copy_from_host(dev.gpu.stream(), row(xs, p))?;
            dev.words.set(dev.gpu.stream(), p as u32, 1)?;
            graph.launch(dev.gpu.stream())?;
            let o = dev.read_cap()?;
            let want = row(outs_s, p);
            let equal = o
                .iter()
                .zip(want)
                .filter(|(a, b)| a.to_bits() == b.to_bits())
                .count();
            ok &= clause(
                &format!("replay p={p}"),
                equal == o.len(),
                &format!(
                    "a captured step's output against the eager step's, {equal} of {} values equal",
                    o.len()
                ),
            );
        }
        ok &= dev.clean("replay")?;
        Ok(ok)
    }

    /// Clause 6: the output gate: the output follows the gated rule, not the ungated one.
    fn gate_rule(outs_s: &[f32], xs: &[f32], rule: &Rule) -> Result<bool, GateError> {
        let mut ok = true;
        for p in READ_AT {
            let q = rule.query(p, row(xs, p));
            let keys = rule.select(&q, p).0.unwrap_or_else(|| all_keys(p));
            let gated = rule.output(&q, &keys);
            let ungated = rule.output_ungated(&q, &keys);
            let have = row(outs_s, p);
            let (g, u) = (max_rel_err(have, &gated)?, max_rel_err(have, &ungated)?);
            ok &= clause(
                &format!("gate p={p}"),
                g <= BAND && u > 10.0 * BAND,
                &format!("the gated rule {g:.3e} (band {BAND:.0e}), the ungated {u:.3e}"),
            );
        }
        Ok(ok)
    }

    /// Clause 7: every refusal, by name, before any launch.
    fn refusals(
        dev: &mut Dev,
        w: &Weights,
        hp: &Hparams,
        split: &Split,
        decls: &[Decl],
        dir: &Path,
    ) -> Result<bool, GateError> {
        let stream = dev.gpu.stream();
        let mut ok = true;
        let mut refused = |name: &str, r: Result<(), GpuError>, want: &str| {
            let text = r.err().map_or_else(String::new, |e| e.to_string());
            ok &= clause(
                &format!("refusal {name}"),
                text.contains(want),
                &format!("{text:?}"),
            );
        };
        // A mis-shaped site: the fused tensor right, `attn_q_a` of 1,000 rows.
        let kv_b = find(decls, &names::attn_kv_b(LAYER))?;
        let bad = [
            as_decl(kv_b),
            decl(
                names::attn_q_a(LAYER),
                HIDDEN,
                Q_LORA - 24,
                GgmlType::FP8_B128,
                tensor(SEED + 1, Q_LORA - 24, HIDDEN, GgmlType::FP8_B128),
            ),
        ];
        let path = dir.join("ernie55-mla-bad.gguf");
        write_gguf(
            &path,
            &[],
            &bad.iter()
                .map(|d| (d.name.clone(), d.dims.clone(), d.ty, d.bytes.clone()))
                .collect::<Vec<_>>(),
        )?;
        let bad = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let w_bad = Weights::load_where(stream, &bad, |_| false)?;
        refused(
            "a mis-shaped site",
            Mla::load(&dev.gpu, &bad, &w_bad, hp, CTX, &[LAYER]).map(|_| ()),
            &format!("{} has dims", names::attn_q_a(LAYER)),
        );
        drop(bad);
        let _ = std::fs::remove_file(&path);
        // The layers.
        let w_none = Weights::load_where(stream, split, |_| false)?;
        refused(
            "an HCA layer",
            Mla::load(&dev.gpu, split, &w_none, hp, CTX, &[LAYER - 1]).map(|_| ()),
            "is an HCA layer",
        );
        refused(
            "a layer past the file's",
            Mla::load(&dev.gpu, split, &w_none, hp, CTX, &[hp.n_layer + 1]).map(|_| ()),
            "past the file's",
        );
        refused(
            "a layer twice",
            Mla::load(&dev.gpu, split, &w_none, hp, CTX, &[LAYER, LAYER]).map(|_| ()),
            "twice",
        );
        refused(
            "no layer",
            Mla::load(&dev.gpu, split, &w_none, hp, CTX, &[]).map(|_| ()),
            "no layer",
        );
        // The shape, the state and the unit.
        let mut hp2 = Hparams { ..hp.clone() };
        hp2.mla.kv_lora = 256;
        refused(
            "a latent tail the flash does not run",
            Shape::of(&hp2, CTX).map(|_| ()),
            "the flash's latent tail",
        );
        let mut hp2 = Hparams { ..hp.clone() };
        hp2.hca.window = WINDOW + 1;
        refused(
            "a window past the window pass's block",
            Shape::of(&hp2, CTX).map(|_| ()),
            "the forced window's pass runs",
        );
        refused(
            "a state past the shape's rows",
            dev.mla.state(stream, 0, CTX + 1).map(|_| ()),
            "a state of",
        );
        refused(
            "a state of no row",
            dev.mla.state(stream, 0, 0).map(|_| ()),
            "a state of",
        );
        refused(
            "a layer's index past the layers",
            dev.mla.state(stream, 1, CTX).map(|_| ()),
            "layers were loaded",
        );
        refused(
            "a unit too wide",
            dev.mla.scratch(stream, BY_NAME_COLS + 1).map(|_| ()),
            "columns a unit",
        );
        // The span and the chunk: a dense state the chain never reaches (every call refuses first).
        let mut st = dev.mla.state(stream, 0, DENSE_CTX)?;
        let mut words = dev.mla.words(stream, MAX_T)?;
        let x = DeviceBuffer::zeroed(stream, MAX_T * HIDDEN)?;
        let mut o = DeviceBuffer::zeroed(stream, MAX_T * HIDDEN)?;
        let x1 = DeviceBuffer::zeroed(stream, HIDDEN)?;
        let mut o1 = DeviceBuffer::zeroed(stream, HIDDEN)?;
        {
            let (gpu, mla, vha) = (&dev.gpu, &dev.mla, &dev.vha);
            let (bn, s) = (&mut dev.bn, &mut dev.s);
            macro_rules! call {
                () => {
                    Call {
                        gpu,
                        w,
                        vha,
                        words: &words,
                        bn: &mut *bn,
                        s: &mut *s,
                    }
                };
            }
            refused(
                "a span of no token",
                words.set(stream, 0, 0),
                "a span of 0 tokens",
            );
            refused(
                "a span past its words",
                words.set(stream, 0, MAX_T + 1),
                "a span of",
            );
            refused(
                "a span past u32",
                words.set(stream, u32::MAX, 1),
                "past u32",
            );
            words.set(stream, 5, 4)?;
            refused(
                "a prompt off its span",
                mla.enqueue_prompt(&mut call!(), &mut st, 6, 4, &x, &mut o),
                "set the span first",
            );
            refused(
                "a step off its span",
                mla.enqueue_step(&mut call!(), &mut st, 5, &x1, &mut o1),
                "set the span to (pos, 1) first",
            );
            refused(
                "a prompt of no token",
                mla.enqueue_prompt(&mut call!(), &mut st, 5, 0, &x, &mut o),
                "a prompt of no token",
            );
            words.set(stream, DENSE_CTX as u32, 1)?;
            refused(
                "a step at the state's last row + 1",
                mla.enqueue_step(&mut call!(), &mut st, DENSE_CTX as u32, &x1, &mut o1),
                "a step at position",
            );
            words.set(stream, (DENSE_CTX - 2) as u32, 4)?;
            refused(
                "rows past the state",
                mla.enqueue_prompt(&mut call!(), &mut st, (DENSE_CTX - 2) as u32, 4, &x, &mut o),
                "land past the state's",
            );
            words.set(stream, 0, 2)?;
            let short = DeviceBuffer::zeroed(stream, HIDDEN / 2)?;
            refused(
                "a short x",
                mla.enqueue_prompt(&mut call!(), &mut st, 0, 2, &short, &mut o),
                "x holds",
            );
        }
        // The key kernel's turned span.
        let kk = IndexKeyKernels::load(dev.gpu.context())?;
        let x128 = DeviceBuffer::zeroed(stream, IDX_D)?;
        let ones = DeviceBuffer::from_host(stream, &vec![1.0f32; IDX_D])?;
        let pos = DeviceBuffer::from_host(stream, &[0u32])?;
        let cs = DeviceBuffer::zeroed(stream, ROPE)?;
        let mut keys = DeviceTensor::zeroed(stream, 4, IDX_D)?;
        refused(
            "a turned span that is not two lanes' pairs",
            kk.enqueue_index_key_ln_neox(
                stream,
                IndexKeyLnNeoxArgs {
                    x: &x128,
                    gain: &ones,
                    bias: &ones,
                    pos: &pos,
                    cs: &cs,
                    eps: EPS,
                    n_dims: 12,
                    m: 1,
                    cache: &mut keys,
                    fault: dev.gpu.unlabelled_sink(),
                },
            ),
            "the turned span",
        );
        stream.synchronize()?;
        ok &= dev.clean("refusals")?;
        Ok(ok)
    }

    /// Clause 8: the faults: each site, raised in its layer alone.
    fn faults(dev: &mut Dev, w: &Weights, xs: &[f32]) -> Result<Vec<bool>, GateError> {
        let mut out = Vec::new();
        let (cache_value, cache_pos, f32_product, key_count) = (
            1 << FaultSite::CacheValue as u32,
            1 << FaultSite::CachePos as u32,
            1 << FaultSite::F32Product as u32,
            FaultSite::KeyCount as u32,
        );
        // A NaN input on a selecting state, so both key appends run: the latent append's norm
        // raises `F32Product` (`fused::kv_norm_rope_append`: a value that is not finite; its
        // `CacheValue` is a finite value past f16), the indexer key's `CacheValue`, both in layer
        // 7's mask alone, and no output is a plausible value.
        let mut st = dev.mla.state(dev.gpu.stream(), 0, CTX)?;
        let mut x = xs[..HIDDEN].to_vec();
        x[0] = f32::NAN;
        let o = dev.step(w, &mut st, 0, &x)?;
        let f = dev.gpu.take_fault()?;
        let stream = dev.gpu.stream();
        let ctx = dev.gpu.context();
        out.push(clause(
            "fault NaN input",
            f.as_ref().is_some_and(|f| {
                f.layer as usize == LAYER
                    && f.code == FaultSite::CacheValue as u32
                    && f.sites == cache_value | f32_product
            }) && o.iter().all(|v| !v.is_finite()),
            &format!(
                "{f:?}, {} of {} outputs finite",
                o.iter().filter(|v| v.is_finite()).count(),
                o.len()
            ),
        ));
        // A position past the cache: the key append raises CachePos and writes nothing.
        let fused = FusedKernels::load(ctx)?;
        let kv_a = DeviceBuffer::zeroed(stream, ROW)?;
        let gain = DeviceBuffer::from_host(stream, &vec![1.0f32; KV_LORA])?;
        let cs = DeviceBuffer::zeroed(stream, ROPE)?;
        let pos = DeviceBuffer::from_host(stream, &[16u32])?;
        let (mut kv_s, mut kvr) = (
            DeviceBuffer::zeroed(stream, ROW)?,
            DeviceBuffer::zeroed(stream, ROW)?,
        );
        let mut cache = DeviceTensor::zeroed(stream, 16, ROW)?;
        fused.enqueue_kv_norm_rope_append(
            stream,
            &kv_a,
            &gain,
            &cs,
            &pos,
            EPS,
            KV_LORA,
            ROPE,
            &mut kv_s,
            &mut kvr,
            &mut cache,
            dev.gpu.layer_sink(LAYER)?,
        )?;
        stream.synchronize()?;
        let f = dev.gpu.take_fault()?;
        out.push(clause(
            "fault key append past the cache",
            f.as_ref()
                .is_some_and(|f| f.layer as usize == LAYER && f.code == FaultSite::CachePos as u32)
                && cache.buf().to_host_vec(stream)?.iter().all(|&v| v == 0),
            &format!("{f:?}, the cache untouched"),
        ));
        // The index-key entry: the same site, the host twin's bit.
        let kk = IndexKeyKernels::load(ctx)?;
        let x128 = DeviceBuffer::zeroed(stream, IDX_D)?;
        let ones = vec![1.0f32; IDX_D];
        let ones_d = DeviceBuffer::from_host(stream, &ones)?;
        let pos = DeviceBuffer::from_host(stream, &[4u32])?;
        let mut keys = DeviceTensor::zeroed(stream, 4, IDX_D)?;
        kk.enqueue_index_key_ln_neox(
            stream,
            IndexKeyLnNeoxArgs {
                x: &x128,
                gain: &ones_d,
                bias: &ones_d,
                pos: &pos,
                cs: &cs,
                eps: EPS,
                n_dims: ROPE,
                m: 1,
                cache: &mut keys,
                fault: dev.gpu.layer_sink(LAYER)?,
            },
        )?;
        stream.synchronize()?;
        let f = dev.gpu.take_fault()?;
        let mut host = vec![0.0f32; 4 * IDX_D];
        let sites = index_key_ln_neox_host(
            &vec![0.0f32; IDX_D],
            (&ones, &ones),
            &[4],
            &vec![0.0f32; ROPE],
            EPS,
            ROPE,
            &mut host,
        );
        out.push(clause(
            "fault index key past the cache",
            f.as_ref()
                .is_some_and(|f| f.layer as usize == LAYER && f.code == FaultSite::CachePos as u32)
                && sites & cache_pos != 0
                && keys.buf().to_host_vec(stream)?.iter().all(|&v| v == 0.0),
            &format!("{f:?}, the twin's site mask {sites:#x}, the keys untouched"),
        ));
        // A window wider than its block: KeyCount, the list and the count untouched.
        let ix = IndexerF32Kernels::load(ctx)?;
        let ints = DeviceBuffer::from_host(stream, &[0u32, 200, TOP_K as u32])?;
        let mut list = DeviceBuffer::from_host(stream, &vec![LIST_POISON; TOP_K + WINDOW])?;
        let mut counts = DeviceBuffer::from_host(stream, &[LIST_POISON])?;
        ix.enqueue_window(
            stream,
            WindowArgs {
                ints: &ints,
                n_vis_at: 0,
                n_win_at: 1,
                top_k_at: 2,
                tokens: 1,
                stride: TOP_K + WINDOW,
                fault: dev.gpu.unlabelled_sink(),
                list: &mut list,
                counts: &mut counts,
            },
        )?;
        stream.synchronize()?;
        let f = dev.gpu.take_fault()?;
        out.push(clause(
            "fault window past its block",
            f.as_ref().is_some_and(|f| f.code == key_count)
                && list.to_host_vec(stream)?.iter().all(|&v| v == LIST_POISON)
                && counts.to_host_vec(stream)?[0] == LIST_POISON,
            &format!("{f:?}, the list and the count untouched"),
        ));
        dev.gpu.clear_fault()?;
        Ok(out)
    }

    // ------------------------------------------------------------------ run

    pub fn run() -> Result<(), GateError> {
        let gpu = Gpu::new()?;
        println!(
            "{NAME}: device {} — heads={HEADS} nope={NOPE}+{ROPE} v={HEAD_V} kv_lora={KV_LORA} \
             q_lora={Q_LORA} hidden={HIDDEN}, indexer {IDX_HEADS}x{IDX_D} top_k={TOP_K} \
             window={WINDOW}, layer {LAYER}, ctx {CTX} and {DENSE_CTX}",
            gpu.device_name()?
        );
        gpu.clear_fault()?;
        let mut ok = e4m3_table_agrees()?;
        let dir = std::env::temp_dir().join(format!("bloomery-ernie55-mla-{}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        let outcome = clauses(gpu, &dir);
        let cleanup = std::fs::remove_dir_all(&dir).map_err(|e| format!("cleanup: {e}"));
        ok &= outcome?;
        cleanup?;
        println!("{NAME}: {}", tier::tally_line());
        if !ok {
            return Err(checks_failed());
        }
        println!(
            "PASSED: {NAME} the k_b/v_b repack is the host split's bytes within its Q8_0 half step, \
             the dense and the selected step match the f64 rule with the indexer's list exact, the \
             prompt form is the step's bits, a captured step replays the eager step's bits, the \
             output gate holds, and the refusals and the faults name themselves"
        );
        Ok(())
    }

    fn clauses(gpu: Gpu, dir: &Path) -> Result<bool, GateError> {
        let stream = gpu.stream();
        let decls = decls();
        let path = dir.join("ernie55-mla.gguf");
        write_gguf(
            &path,
            &[],
            &decls
                .iter()
                .map(|d| (d.name.clone(), d.dims.clone(), d.ty, d.bytes.clone()))
                .collect::<Vec<_>>(),
        )?;
        let split = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let hp = hparams();
        let prefix = format!("blk.{LAYER}.");
        let mut w = Weights::load_where(stream, &split, |n| n.starts_with(&prefix))?;
        derive(stream, &split, &hp, [LAYER], &mut w)?;
        let mla = Mla::load(&gpu, &split, &w, &hp, CTX, &[LAYER])?;
        let halves = split_kv_b(&mla.shape(), &split, &Sites::of(LAYER))?;
        // The rule: the file's weights, the derived halves, the key caches for every position.
        let f = |n: &str| find(&decls, n).and_then(rows_of);
        let kv_b = f(&names::attn_kv_b(LAYER))?;
        let w_f = WeightsF {
            q_a: f(&names::attn_q_a(LAYER))?,
            q_b: f(&names::attn_q_b(LAYER))?,
            kv_a: f(&names::attn_kv_a_mqa(LAYER))?,
            gate: f(&names::attn_gate(LAYER))?,
            o: f(&names::attn_output(LAYER))?,
            idx_q: f(&names::indexer_attn_q_b(LAYER))?,
            idx_k: f(&names::indexer_attn_k(LAYER))?,
            idx_w: f(&names::indexer_proj(LAYER))?,
            q_a_norm: f(&names::attn_q_a_norm(LAYER))?,
            kv_a_norm: f(&names::attn_kv_a_norm(LAYER))?,
            sinks: f(&names::attn_sinks(LAYER))?,
            kn_w: f(&names::indexer_k_norm(LAYER))?,
            kn_b: f(&names::indexer_k_norm_bias(LAYER))?,
            k_b: q8_rows(&halves.parts[0].1, NOPE)?,
            v_b: q8_rows(&halves.parts[1].1, KV_LORA)?,
        };
        let xs = inputs();
        let rule = Rule::new(w_f, &xs)?;
        let bn = mla.forms(&gpu, &w, UNIT)?;
        let s = mla.scratch(stream, UNIT)?;
        let words = mla.words(stream, MAX_T)?;
        let (xs_dev, x1, o1) = (
            DeviceBuffer::from_host(stream, &xs)?,
            DeviceBuffer::zeroed(stream, HIDDEN)?,
            DeviceBuffer::zeroed(stream, HIDDEN)?,
        );
        let (oc, x_cap, o_cap) = (
            DeviceBuffer::zeroed(stream, MAX_T * HIDDEN)?,
            DeviceBuffer::zeroed(stream, HIDDEN)?,
            DeviceBuffer::zeroed(stream, HIDDEN)?,
        );
        let mut dev = Dev {
            gpu,
            mla,
            bn,
            s,
            words,
            vha: NoVha,
            xs: xs_dev,
            x1,
            o1,
            oc,
            x_cap,
            o_cap,
        };
        let mut ok = true;
        tier::sc(
            "1 repack: the derived halves are the host split's bytes within the Q8_0 half step",
        )?;
        ok &= repack(&dev.gpu, &w, &halves, &kv_b)?;
        tier::sc(
            "2 dense: the step below the cut against the f64 rule, the cache rows and keys against the rule's",
        )?;
        let mut st_d = dev.mla.state(dev.gpu.stream(), 0, DENSE_CTX)?;
        ok &= dense(&mut dev, &w, &mut st_d, &xs, &rule)?;
        tier::sc(
            "3 selected: the step above the cut, the list exact against the rule, the output against the rule over it",
        )?;
        let mut st_s = dev.mla.state(dev.gpu.stream(), 0, CTX)?;
        let (sel_ok, outs_s) = selected(&mut dev, &w, &mut st_s, &xs, &rule)?;
        ok &= sel_ok;
        tier::sc("4 prompt: the prompt chunks' outputs and cache rows are the step form's bits")?;
        ok &= prompt(&mut dev, &w, &xs, &outs_s)?;
        tier::sc("5 replay: a captured step replayed at other positions is the eager step's bits")?;
        ok &= replay(&mut dev, &w, &xs, &outs_s)?;
        tier::sc("6 gate: the output follows the gated rule and not the ungated one")?;
        ok &= gate_rule(&outs_s, &xs, &rule)?;
        tier::sc(
            "7 refusals: every shape the chain does not run is refused by name before a launch",
        )?;
        ok &= refusals(&mut dev, &w, &hp, &split, &decls, dir)?;
        tier::sc(
            "8 faults: the NaN input, the position past the cache and the window past its block raise their sites",
        )?;
        for pass in faults(&mut dev, &w, &xs)? {
            ok &= pass;
        }
        tier::sc("9 shape: the three new entries compile with no local depot")?;
        ok &= no_local_depot(&["index_key_ln_neox", "indexer_score_f32", "indexer_window"])?;
        Ok(ok)
    }
}
