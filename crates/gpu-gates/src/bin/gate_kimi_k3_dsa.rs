//! GPU gate for Kimi K3.1's selected attention on the card, model-less at the released file's
//! shapes (the lightning indexer's 32 heads of 128 and top-k 2048 over a 4096-row cache; the gated
//! latent layer at 2 heads) against host rules and an f64 reference written from llama.cpp's
//! `build_dsa_layer` in `src/models/kimi-k31.cpp`. Inputs are seeded; nothing is read from a model
//! file.
//!
//! The step is `bloomery_gpu::arch::kimi_k3`'s: [`enqueue_keys`] (the index key of the row's key
//! projection: a LayerNorm with gain and bias at `attention.layer_norm_epsilon`, no rope, the
//! 128-wide Hadamard transform, f16 into the key row at the position), [`enqueue_pick`] (the score
//! `Σ_h w_h · relu(q_h · k)` of every visible row with the weights scaled by `1/√(32·128)`, and
//! the `min(pos + 1, top_k)` best rows in ascending order) over the count words of
//! [`count_words`], then `latent_gated::attend_sel` over the list (the gather and the latent
//! attention over the gathered rows). The projections that feed the step (`indexer.attn_k`,
//! `indexer.attn_q_b`, `indexer.proj` and the latent layer's) are dense products other gates own;
//! here their outputs are the seeded inputs, as the graph names them. The file's graph applies no
//! transform; the transform is orthogonal, so the graph's score is the same dot in another basis,
//! and the host computes the graph's score from the untransformed query and key.
//!
//! One run steps positions 0 .. 4096 on the selected path and, below top-k, a dense `attend` on
//! a second cache of the same height in lockstep.
//!
//! 1. `key`: every key row the steps wrote equals the host transcription of the key kernel
//!    (`index_key`'s module doc: the LayerNorm's f64 lane sums, the f32 ops, `fast_ht`, f16) bit
//!    for bit; a rows-pass launch of 8 rows writes the same bits at its positions and nothing
//!    beside them.
//! 2. `list`: at positions 2048, 2049, 3000 and 4095 the score pass's query is `fast_ht` of the
//!    query bit for bit and its weights `w · 2⁻⁶` bit for bit; the list is the exact top-k of the
//!    card's scores, ascending, ties to the lower row, bit for bit; every score lies within the
//!    kernel's band of our rule (the split query's exact f16 products summed in f64); the list
//!    agrees with our rule's top-k and with the graph's (the f64 score of the untransformed query
//!    and the f64 LayerNorm key) at every row outside the tie band. At 4095 the lists of the graph
//!    without the head weights and without the relu each disagree past 1 % of the rows.
//! 3. `identity`: at every position below top-k the selected step's gated heads equal the dense
//!    step's bit for bit, through the real score and top-k passes; the list is `0 ..= pos` at
//!    positions 0, 63, 64 and 2047.
//! 4. `selected`: at the positions of clause 2 the gated heads lie within [`BAND`] of the graph
//!    over the card's list; at 4095 the graph over every key and the graph over the first top-k
//!    rows each lie past [`SEP`] bands of the card.
//! 5. `fault`: a list entry at the cache's height raises `CachePos` from the gather; a key
//!    projection holding a NaN raises `CacheValue`, and a key position at the key rows' height
//!    `CachePos`, from the key launch; each read with the launch's layer and cleared after.
//! 6. `refusals`: a selection on a file with no indexer, a list of another width than top-k, a
//!    position at u32's last value, and every selection shape `attend_sel` does not take are
//!    refused by name.
//! 7. `pass`: a pass of [`ACROSS_M`] rows from position [`ACROSS_AT`], across top-k (its first
//!    rows' lists the identity, its last rows' a pick of top-k of more rows), over a cache and key
//!    rows holding the steps' rows below the pass and an f16 NaN from it: the rows' count words
//!    from [`pass_words`], every row's key row from one launch of [`enqueue_keys`] at the pass's
//!    rows, then [`attend_rows_sel`] (each row's pick and selected core in position order). Every
//!    row's gated heads, and the key rows and cache rows the pass wrote, are the steps' bits; a
//!    pass whose count words hold fewer rows than the pass is refused by name.
//!
//! The indexer's score pass has no fault word: a NaN query or weight scores as `relu` maps it, to
//! 0, so this gate has no clause for one; a NaN key is raised by the key launch (clause 5).

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_kimi_k3_dsa", gate::run())
}

#[cfg(feature = "gpu")]
mod gate {
    use bloomery_gpu::arch::kimi_k3::{
        IndexKernels, N_SEL_AT, PassPick, WORDS, attend_rows_sel, check_selection, count_words,
        enqueue_keys, enqueue_pick, pass_words,
    };
    use bloomery_gpu::indexer::{HEAD_DIM, HEADS as IX_HEADS, IndexerScratch};
    use bloomery_gpu::latent::LATENT;
    use bloomery_gpu::latent_gated::{
        AttendIn, AttnBufs, MlaDims, MlaKernels, Selection, ValueW, attend, attend_sel,
        partials_lens,
    };
    use bloomery_gpu::weights::q8_0_planes;
    use bloomery_gpu::{DeviceTensor, FaultSite, Gpu, GpuError, LAYER_NONE, Window};
    use bloomery_gpu_gates::rounding::{U, butterfly, gamma};
    use bloomery_gpu_gates::{GateError, GateRng, bits_equal, checks_failed, par_map, verdict};
    use cuda_core::DeviceBuffer;
    use gguf::quant::{Q8Block, f32_to_f16_bits, half_to_f32};
    use model::arch::kimi_k3::hparams::Indexer;

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
    /// `attention.layer_norm_rms_epsilon`, the latent layer's norms'.
    const EPS: f32 = 1e-5;
    /// `attention.layer_norm_epsilon`, the index key's LayerNorm's (the fixture header's).
    const IX_EPS: f32 = 1e-5;
    /// The latent layer's query heads: the selection is the same for every head count.
    const HEADS: usize = 2;
    /// The released K3.1 file's `attention.indexer.top_k`.
    const TOP_K: usize = 2048;
    /// The cache's rows and the positions the run steps.
    const CTX: usize = 4096;
    /// The weights' scale `1/√(32·128)`, exact in f32 and f64.
    const W_SCALE: f64 = 1.0 / 64.0;
    /// The selected attention's band against the graph: the f16 cache and the f16 query rows of
    /// the tensor-core pass each move a score or a value by under 5e-4 relative; 4e-3 is eight
    /// times that (`gate_kimi_k3_mla`'s band, the same arithmetic over other rows).
    const BAND: f64 = 4e-3;
    /// The bands a reference missing a clause must lie past.
    const SEP: f64 = 10.0;
    /// Positions whose list the identity clause reads.
    const IDENTITY_LISTS: [usize; 4] = [0, 63, 64, TOP_K - 1];
    /// Positions past top-k whose list, scores and heads the run saves.
    const SAVES: [usize; 4] = [TOP_K, TOP_K + 1, 3000, CTX - 1];
    /// The tensor-core dot's bound in units of `U` times its products' magnitude [derived]: an
    /// `mma` forms every f16 product exactly and adds its sixteen products and the accumulator
    /// with each addend cut at the 24th bit of the largest and the sum rounded or cut once more,
    /// so one k-step errs by at most 18·2⁻²³ = 36u of its addends' magnitude; the eight k-steps of
    /// a head err by at most 8·36u = 288u of the head's product magnitude.
    const TC_UNITS: f64 = 288.0;
    /// Roundings one head's weighted term meets in the kernel's head sum: a product and three
    /// fused multiply-adds along its lane group, then three butterfly levels.
    const HEAD_SUM_ROUNDINGS: usize = 7;
    /// The rows-pass key launch: its first position and its rows.
    const PASS_AT: usize = 100;
    const PASS_M: usize = 8;
    /// What a key row holds before a launch writes it: an f16 NaN no key the launch writes is.
    const KEY_FILL: u16 = 0x7e01;
    /// The pass clause's first position and rows: across top-k, half its rows below it.
    const ACROSS_AT: usize = TOP_K - ACROSS_M / 2;
    const ACROSS_M: usize = 8;
    /// The fault clause's cache, list width and position.
    const F_CTX: usize = 128;
    const F_TOP_K: usize = 64;

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

    /// The latent layer's widths; `attend` reads the heads, the nope, the rope lanes and the value
    /// width. The rest are the file's.
    fn dims() -> MlaDims {
        MlaDims {
            embd: 7168,
            heads: HEADS,
            q_lora: 1536,
            nope: NOPE,
            extra: EXTRA,
            v_head: V_HEAD,
            rms_eps: EPS,
            latent_eps: EPS,
        }
    }

    // ------------------------------------------------------------ the layer's weights and rows

    /// Q8_0 planes as the card holds them: the quants as words, the f16 scales.
    type Q8Planes = (Vec<u32>, Vec<u16>);

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

    /// One layer's weights and `CTX` positions of its projections' outputs: the latent layer's
    /// (as `gate_kimi_k3_mla` seeds them) and the indexer's.
    struct Layer {
        /// `attn_k_b`, Q8_0: `[heads][LATENT][NOPE]`.
        kb_planes: Q8Planes,
        kb: Vec<f64>,
        /// `attn_v_b`, Q8_0: `[heads][V_HEAD][LATENT]`.
        vb_planes: Q8Planes,
        vb: Vec<f64>,
        /// `attn_kv_a_norm`.
        gain: Vec<f32>,
        /// `[CTX][ROW]`: `attn_kv_a_mqa`'s output, the latent then the rope lanes.
        kv: Vec<f32>,
        /// `[CTX][heads·HEAD_K]`: `attn_q_b`'s output.
        q: Vec<f32>,
        /// `[CTX][heads·V_HEAD]`: `attn_gate`'s output.
        gate: Vec<f32>,
        /// `indexer.k_norm`'s gain and bias.
        ix_gain: Vec<f32>,
        ix_bias: Vec<f32>,
        /// `[CTX][HEAD_DIM]`: `indexer.attn_k`'s output.
        ik: Vec<f32>,
        /// `[CTX][IX_HEADS·HEAD_DIM]`: `indexer.attn_q_b`'s output.
        iq: Vec<f32>,
        /// `[CTX][IX_HEADS]`: `indexer.proj`'s output.
        iw: Vec<f32>,
    }

    impl Layer {
        fn new(seed: u64) -> Result<Layer, GateError> {
            let mut r = GateRng(seed);
            let (kb_planes, kb) = q8(&fill(&mut r, HEADS * LATENT * NOPE, -1.0, 1.0))?;
            let (vb_planes, vb) = q8(&fill(&mut r, HEADS * V_HEAD * LATENT, -1.0, 1.0))?;
            let mut kv = Vec::with_capacity(CTX * ROW);
            for _ in 0..CTX {
                kv.extend(fill(&mut r, LATENT, -1.0, 1.0));
                kv.extend(fill(&mut r, EXTRA, -2.0, 2.0));
            }
            let mut q = Vec::with_capacity(CTX * HEADS * HEAD_K);
            for _ in 0..CTX * HEADS {
                q.extend(fill(&mut r, NOPE, -0.25, 0.25));
                q.extend(fill(&mut r, EXTRA, -2.5, 2.5));
            }
            Ok(Layer {
                kb_planes,
                kb,
                vb_planes,
                vb,
                gain: fill(&mut r, LATENT, 0.5, 1.5),
                kv,
                q,
                gate: fill(&mut r, CTX * HEADS * V_HEAD, -3.0, 3.0),
                ix_gain: fill(&mut r, HEAD_DIM, 0.5, 1.5),
                ix_bias: fill(&mut r, HEAD_DIM, -0.25, 0.25),
                ik: fill(&mut r, CTX * HEAD_DIM, -2.0, 2.0),
                iq: fill(&mut r, CTX * IX_HEADS * HEAD_DIM, -1.0, 1.0),
                iw: fill(&mut r, CTX * IX_HEADS, -1.0, 1.0),
            })
        }

        fn ik_row(&self, p: usize) -> &[f32] {
            &self.ik[p * HEAD_DIM..(p + 1) * HEAD_DIM]
        }
    }

    // ------------------------------------------------------------------ the index key rules

    /// The key kernel's LayerNorm of one key, op for op (`index_key`'s module doc): the lane sums
    /// (lane `l` the values `4l ..`) in f64, the mean rounded to f32, the f32 deviations, the f32
    /// squares added in f64, the variance, the scale, then `((dev · scale) · gain) + bias` each
    /// rounded.
    fn key_norm(x: &[f32], gain: &[f32], bias: &[f32]) -> Vec<f32> {
        let lanes = |f: &dyn Fn(usize) -> f64| -> [f64; 32] {
            std::array::from_fn(|l| (0..4).fold(0.0f64, |a, j| a + f(4 * l + j)))
        };
        let mean = (butterfly(lanes(&|i| f64::from(x[i]))) / 128.0) as f32;
        let dev: Vec<f32> = x.iter().map(|&v| v - mean).collect();
        let var = (butterfly(lanes(&|i| f64::from(dev[i] * dev[i]))) / 128.0) as f32;
        let scale = if var.is_finite() {
            1.0 / (var + IX_EPS).sqrt()
        } else {
            f32::NAN
        };
        (0..HEAD_DIM)
            .map(|i| ((dev[i] * scale) * gain[i]) + bias[i])
            .collect()
    }

    /// ik's `fast_ht`, the kernels' transform: butterflies at `h = 1, 2, 4, …`, `x + y` and
    /// `x − y`, the scale multiplied by the f32 `FRAC_1_SQRT_2` per stage, then every value by it.
    fn fast_ht(v: &mut [f32]) {
        let n = v.len();
        let mut scale = 1.0f32;
        let mut h = 1;
        while h < n {
            for i in (0..n).step_by(2 * h) {
                for j in i..i + h {
                    let (x, y) = (v[j], v[j + h]);
                    v[j] = x + y;
                    v[j + h] = x - y;
                }
            }
            scale *= std::f32::consts::FRAC_1_SQRT_2;
            h <<= 1;
        }
        for x in v.iter_mut() {
            *x *= scale;
        }
    }

    /// The key kernel's row for one key: the LayerNorm, the transform, f16.
    fn key_host(x: &[f32], gain: &[f32], bias: &[f32]) -> Vec<u16> {
        let mut v = key_norm(x, gain, bias);
        fast_ht(&mut v);
        v.iter().map(|&e| f32_to_f16_bits(e)).collect()
    }

    /// The graph's key: `ggml_norm` then the gain and the bias, in f64, untransformed.
    fn key_graph(x: &[f32], gain: &[f32], bias: &[f32]) -> Vec<f64> {
        let n = x.len() as f64;
        let mean = x.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
        let var = x
            .iter()
            .map(|&v| (f64::from(v) - mean).powi(2))
            .sum::<f64>()
            / n;
        let inv = 1.0 / (var + f64::from(IX_EPS)).sqrt();
        x.iter()
            .zip(gain.iter().zip(bias))
            .map(|(&v, (&g, &b))| (f64::from(v) - mean) * inv * f64::from(g) + f64::from(b))
            .collect()
    }

    // ------------------------------------------------------------------ the score rules

    /// The split of one query value the score kernel makes: `hi = f16(q)`, `lo = f16(q − hi)`.
    fn split(q: f32) -> (f64, f64) {
        let hi = half_to_f32(f32_to_f16_bits(q));
        let lo = half_to_f32(f32_to_f16_bits(q - hi));
        (f64::from(hi), f64::from(lo))
    }

    /// Every visible row's scores at one position.
    struct Scores {
        /// Our rule: `Σ_h w̃_h · relu(Σ_d (hi + lo)·k̃)` over the card's transformed query, its
        /// scaled weights and its f16 key rows, the exact products summed in f64.
        rule: Vec<f64>,
        /// The kernel's distance to `rule`: per head the tensor cores' [`TC_UNITS`] of the
        /// products' magnitude and one rounding of `acc_hi + acc_lo`, carried by `|w̃_h|`; then
        /// the head sum's [`HEAD_SUM_ROUNDINGS`].
        kernel: Vec<f64>,
        /// The graph: `Σ_h w_h·2⁻⁶ · relu(q_h · k)` over the untransformed query and the f64
        /// LayerNorm key.
        graph: Vec<f64>,
        /// The graph without the head weights, and without the relu.
        no_weights: Vec<f64>,
        no_relu: Vec<f64>,
    }

    fn scores(l: &Layer, p: usize, qt: &[f32], wt: &[f32], keys: &[u16]) -> Scores {
        let sp: Vec<(f64, f64)> = qt.iter().map(|&v| split(v)).collect();
        let q_raw = &l.iq[p * IX_HEADS * HEAD_DIM..(p + 1) * IX_HEADS * HEAD_DIM];
        let w_raw = &l.iw[p * IX_HEADS..(p + 1) * IX_HEADS];
        let rows = par_map(p + 1, |t| {
            let kt: Vec<f64> = keys[t * HEAD_DIM..(t + 1) * HEAD_DIM]
                .iter()
                .map(|&b| f64::from(half_to_f32(b)))
                .collect();
            let kg = key_graph(l.ik_row(t), &l.ix_gain, &l.ix_bias);
            let (mut rule, mut kb, mut smag) = (0.0f64, 0.0f64, 0.0f64);
            let (mut graph, mut nw, mut nr) = (0.0f64, 0.0f64, 0.0f64);
            for h in 0..IX_HEADS {
                let (mut dot, mut mag) = (0.0f64, 0.0f64);
                for (&(hi, lo), &k) in sp[h * HEAD_DIM..(h + 1) * HEAD_DIM].iter().zip(&kt) {
                    dot += (hi + lo) * k;
                    mag += (hi * k).abs() + (lo * k).abs();
                }
                let wh = f64::from(wt[h]);
                let relu = dot.max(0.0);
                rule += wh * relu;
                smag += wh.abs() * relu;
                kb += wh.abs() * (TC_UNITS * U * mag + 2.0 * U * dot.abs());
                let g: f64 = q_raw[h * HEAD_DIM..(h + 1) * HEAD_DIM]
                    .iter()
                    .zip(&kg)
                    .map(|(&q, &k)| f64::from(q) * k)
                    .sum();
                let w = f64::from(w_raw[h]) * W_SCALE;
                graph += w * g.max(0.0);
                nw += g.max(0.0);
                nr += w * g;
            }
            let hs = gamma(HEAD_SUM_ROUNDINGS);
            [rule, (1.0 + hs) * kb + hs * smag, graph, nw, nr]
        });
        let col = |i: usize| rows.iter().map(|r| r[i]).collect();
        Scores {
            rule: col(0),
            kernel: col(1),
            graph: col(2),
            no_weights: col(3),
            no_relu: col(4),
        }
    }

    /// A distance past its band; a NaN distance is past every band.
    fn outside(d: f64, band: f64) -> bool {
        d.is_nan() || d > band
    }

    /// The order-preserving key of a score — the kernels' `order_key`, transcribed: `-0` made
    /// `+0`, a negative's bits flipped whole, a non-negative's top bit set.
    fn key_of(v: f32) -> u32 {
        let b = v.to_bits();
        let b = if b == 0x8000_0000 { 0 } else { b };
        if b & 0x8000_0000 != 0 {
            !b
        } else {
            b | 0x8000_0000
        }
    }

    /// The `k` rows of `scores` with the largest keys, ties to the lower row, in ascending row
    /// order: what the top-k pass must write, bit for bit.
    fn exact_top(scores: &[f32], k: usize) -> Vec<u32> {
        let mut idx: Vec<u32> = (0..scores.len() as u32).collect();
        idx.sort_by(|&a, &b| {
            key_of(scores[b as usize])
                .cmp(&key_of(scores[a as usize]))
                .then(a.cmp(&b))
        });
        let mut top = idx[..k].to_vec();
        top.sort_unstable();
        top
    }

    /// A list `ids` against the top-k of `score` at every row outside the tie band: row `t` is
    /// left out when `|s_t − s_k| <= β_t + β_k`, `s_k` the k-th score. The rows that disagree and
    /// those left out.
    fn ids_vs(ids: &[u32], score: &[f64], beta: &[f64], k: usize) -> (usize, usize) {
        let mut order: Vec<usize> = (0..score.len()).collect();
        order.sort_by(|&a, &b| score[b].total_cmp(&score[a]).then(a.cmp(&b)));
        let kth = order[k - 1];
        let (sk, bk) = (score[kth], beta[kth]);
        let mut taken = vec![false; score.len()];
        for &i in ids {
            if let Some(t) = taken.get_mut(i as usize) {
                *t = true;
            }
        }
        let (mut off, mut out) = (0usize, 0usize);
        for (t, (&s, &b)) in score.iter().zip(beta).enumerate() {
            if (s - sk).abs() <= b + bk {
                out += 1;
            } else if taken[t] != (s > sk) {
                off += 1;
            }
        }
        (off, out)
    }

    // ------------------------------------------------------------------ the latent graph

    /// The graph's per-position latent keys: the normed latent and `k_pe` as it came.
    struct Keys {
        ln: Vec<Vec<f64>>,
        kpe: Vec<Vec<f64>>,
    }

    fn keys(l: &Layer) -> Keys {
        let mut ln = Vec::with_capacity(CTX);
        let mut kpe = Vec::with_capacity(CTX);
        for t in 0..CTX {
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

    /// The graph for the query at position `p` over the cache rows `rows`, gated:
    /// `[heads·V_HEAD]`.
    fn graph(l: &Layer, ks: &Keys, p: usize, rows: &[usize]) -> Vec<f64> {
        let scale = 1.0 / (HEAD_K as f64).sqrt();
        let per_head = par_map(HEADS, |h| {
            let q = &l.q[(p * HEADS + h) * HEAD_K..(p * HEADS + h + 1) * HEAD_K];
            let qa: Vec<f64> = (0..LATENT)
                .map(|c| {
                    let row = &l.kb[(h * LATENT + c) * NOPE..(h * LATENT + c + 1) * NOPE];
                    row.iter()
                        .zip(&q[..NOPE])
                        .map(|(&a, &b)| a * f64::from(b))
                        .sum()
                })
                .collect();
            let logits: Vec<f64> = rows
                .iter()
                .map(|&t| {
                    let s: f64 = qa.iter().zip(&ks.ln[t]).map(|(a, b)| a * b).sum::<f64>()
                        + q[NOPE..]
                            .iter()
                            .zip(&ks.kpe[t])
                            .map(|(&a, b)| f64::from(a) * b)
                            .sum::<f64>();
                    s * scale
                })
                .collect();
            let mx = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let e: Vec<f64> = logits.iter().map(|s| (s - mx).exp()).collect();
            let z: f64 = e.iter().sum();
            let mut lat = vec![0.0f64; LATENT];
            for (w, &t) in e.iter().zip(rows) {
                for (a, &b) in lat.iter_mut().zip(&ks.ln[t]) {
                    *a += w * b;
                }
            }
            (0..V_HEAD)
                .map(|r| {
                    let row = &l.vb[(h * V_HEAD + r) * LATENT..(h * V_HEAD + r + 1) * LATENT];
                    let v: f64 = row.iter().zip(&lat).map(|(a, b)| a * b).sum::<f64>() / z;
                    v * sigmoid(f64::from(l.gate[(p * HEADS + h) * V_HEAD + r]))
                })
                .collect::<Vec<f64>>()
        });
        per_head.into_iter().flatten().collect()
    }

    // ------------------------------------------------------------------ the card

    /// The layer's resident weights and kernels.
    struct Dev<'a> {
        gpu: &'a Gpu,
        l: &'a Layer,
        d: MlaDims,
        mk: MlaKernels,
        ik: IndexKernels,
        kb: (DeviceTensor<u32>, DeviceTensor<u16>),
        vb: (DeviceTensor<u32>, DeviceTensor<u16>),
        gain: DeviceBuffer<f32>,
        ix_gain: DeviceBuffer<f32>,
        ix_bias: DeviceBuffer<f32>,
    }

    /// One latent run's cache, inputs and scratch.
    struct Mla {
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

    /// One selected run's key rows, indexer inputs, count words, list and gathered rows.
    struct Ix {
        keys: DeviceTensor<u16>,
        ik: DeviceBuffer<f32>,
        iq: DeviceBuffer<f32>,
        iw: DeviceBuffer<f32>,
        words: DeviceBuffer<u32>,
        scratch: IndexerScratch,
        list: DeviceBuffer<u32>,
        gathered: DeviceTensor<u16>,
        top_k: usize,
    }

    /// What the run read back at a saved position.
    struct Saved {
        p: usize,
        y: Vec<f32>,
        list: Vec<u32>,
        qt: Vec<f32>,
        wt: Vec<f32>,
        scores: Vec<f32>,
    }

    /// What the whole run read back.
    struct RunOut {
        /// Positions below top-k whose selected heads were not the dense step's bits.
        not_identical: Vec<usize>,
        /// Positions of [`IDENTITY_LISTS`] whose list was not `0 ..= pos`.
        not_identity_list: Vec<usize>,
        saved: Vec<Saved>,
        keys: Vec<u16>,
        /// The selected run's cache rows, `[CTX][ROW]` f16.
        cache: Vec<u16>,
        /// The selected step's gated heads at each position of the pass clause.
        across: Vec<Vec<f32>>,
    }

    impl<'a> Dev<'a> {
        fn new(gpu: &'a Gpu, l: &'a Layer) -> Result<Dev<'a>, GateError> {
            let s = gpu.stream();
            let d = dims();
            let planes = |(qs, dd): &Q8Planes, rows: usize, k: usize| {
                Ok::<_, GateError>((
                    DeviceTensor::upload(s, qs, rows, k / 4)?,
                    DeviceTensor::upload(s, dd, rows, k / 32)?,
                ))
            };
            Ok(Dev {
                gpu,
                l,
                d,
                mk: MlaKernels::load(gpu, &d, 1)?,
                ik: IndexKernels::load(gpu)?,
                kb: planes(&l.kb_planes, HEADS * LATENT, NOPE)?,
                vb: planes(&l.vb_planes, HEADS * V_HEAD, LATENT)?,
                gain: DeviceBuffer::from_host(s, &l.gain)?,
                ix_gain: DeviceBuffer::from_host(s, &l.ix_gain)?,
                ix_bias: DeviceBuffer::from_host(s, &l.ix_bias)?,
            })
        }

        fn mla(&self, ctx: usize) -> Result<Mla, GateError> {
            let s = self.gpu.stream();
            let hv = HEADS * V_HEAD;
            let z = |n: usize| DeviceBuffer::from_host(s, &vec![0.0f32; n]);
            let (pv, pm) = partials_lens(&self.d, ctx);
            Ok(Mla {
                cache: DeviceTensor::<u16>::zeroed(s, ctx, ROW)?,
                kv: z(ROW)?,
                q: z(HEADS * HEAD_K)?,
                gate: z(hv)?,
                pos: DeviceBuffer::from_host(s, &[0u32])?,
                n_keys: DeviceBuffer::from_host(s, &[0u32])?,
                qabs: z(HEADS * ROW)?,
                att: z(HEADS * LATENT)?,
                av: z(hv)?,
                part_v: z(pv)?,
                part_ms: z(pm)?,
                y: z(hv)?,
            })
        }

        fn ix(&self, ctx: usize, top_k: usize) -> Result<Ix, GateError> {
            let s = self.gpu.stream();
            Ok(Ix {
                keys: DeviceTensor::upload(s, &vec![KEY_FILL; ctx * HEAD_DIM], ctx, HEAD_DIM)?,
                ik: DeviceBuffer::zeroed(s, HEAD_DIM)?,
                iq: DeviceBuffer::zeroed(s, IX_HEADS * HEAD_DIM)?,
                iw: DeviceBuffer::zeroed(s, IX_HEADS)?,
                words: DeviceBuffer::zeroed(s, 3)?,
                scratch: IndexerScratch::new(s, 1, ctx)?,
                list: DeviceBuffer::zeroed(s, top_k)?,
                gathered: DeviceTensor::zeroed(s, top_k, ROW)?,
                top_k,
            })
        }

        /// Stage position `p`'s latent inputs.
        fn stage(&self, m: &mut Mla, p: usize) -> Result<(), GateError> {
            let s = self.gpu.stream();
            let l = self.l;
            let hv = HEADS * V_HEAD;
            m.kv.copy_from_host(s, &l.kv[p * ROW..(p + 1) * ROW])?;
            m.q.copy_from_host(s, &l.q[p * HEADS * HEAD_K..(p + 1) * HEADS * HEAD_K])?;
            m.gate.copy_from_host(s, &l.gate[p * hv..(p + 1) * hv])?;
            let p32 = u32::try_from(p)?;
            m.pos.copy_from_host(s, &[p32])?;
            m.n_keys.copy_from_host(s, &[p32 + 1])?;
            Ok(())
        }

        /// Stage position `p`'s indexer inputs and the step's count words, as the refresh writes
        /// them.
        fn stage_ix(&self, ix: &mut Ix, p: usize, ik: &[f32]) -> Result<(), GateError> {
            let s = self.gpu.stream();
            let l = self.l;
            ix.ik.copy_from_host(s, ik)?;
            ix.iq.copy_from_host(
                s,
                &l.iq[p * IX_HEADS * HEAD_DIM..(p + 1) * IX_HEADS * HEAD_DIM],
            )?;
            ix.iw
                .copy_from_host(s, &l.iw[p * IX_HEADS..(p + 1) * IX_HEADS])?;
            let words = count_words(u32::try_from(p)?, u32::try_from(ix.top_k)?)?;
            ix.words.copy_from_host(s, &words)?;
            Ok(())
        }

        /// The dense step over the staged row.
        fn dense(&self, m: &mut Mla) -> Result<(), GateError> {
            let Mla {
                cache,
                qabs,
                att,
                av,
                part_v,
                part_ms,
                y,
                ..
            } = m;
            let ins = AttendIn {
                kv: &m.kv,
                q: &m.q,
                gate: &m.gate,
                kv_norm: &self.gain,
                k_b: (&self.kb.0, &self.kb.1),
                v_b: ValueW::Q8_0(&self.vb.0, &self.vb.1),
                pos: &m.pos,
                n_keys: &m.n_keys,
                fault: self.gpu.unlabelled_sink(),
            };
            let bufs = AttnBufs {
                qabs,
                att,
                av,
                part_v,
                part_ms,
            };
            attend(self.gpu, &self.mk, &self.d, ins, bufs, cache, y)?;
            Ok(())
        }

        /// The selected step over the staged row: the key, the pick, `poison` written over the
        /// list's head when given, then the latent core over the list.
        fn selected(
            &self,
            m: &mut Mla,
            ix: &mut Ix,
            poison: Option<&[u32]>,
        ) -> Result<(), GpuError> {
            let s = self.gpu.stream();
            enqueue_keys(
                s,
                &self.ik,
                (&self.ix_gain, &self.ix_bias, IX_EPS),
                (&ix.ik, &m.pos, 1),
                self.gpu.unlabelled_sink(),
                &mut ix.keys,
            )?;
            enqueue_pick(
                s,
                &self.ik,
                &ix.keys,
                (&ix.iq, &ix.iw),
                &ix.words,
                &mut ix.scratch,
                &mut ix.list,
            )?;
            if let Some(v) = poison {
                ix.list.copy_from_host(s, v)?;
            }
            let n = Window::<u32>::of(&ix.words, N_SEL_AT * 4, 1)?;
            let sel = Selection {
                list: &ix.list,
                n: &n,
                stride: ix.top_k,
                scratch: &mut ix.gathered,
            };
            self.core(m, sel)
        }

        /// `attend_sel` over the staged row and `sel`.
        fn core(&self, m: &mut Mla, sel: Selection<'_>) -> Result<(), GpuError> {
            let Mla {
                cache,
                qabs,
                att,
                av,
                part_v,
                part_ms,
                y,
                ..
            } = m;
            let ins = AttendIn {
                kv: &m.kv,
                q: &m.q,
                gate: &m.gate,
                kv_norm: &self.gain,
                k_b: (&self.kb.0, &self.kb.1),
                v_b: ValueW::Q8_0(&self.vb.0, &self.vb.1),
                pos: &m.pos,
                n_keys: &m.n_keys,
                fault: self.gpu.unlabelled_sink(),
            };
            let bufs = AttnBufs {
                qabs,
                att,
                av,
                part_v,
                part_ms,
            };
            attend_sel(self.gpu, &self.mk, &self.d, ins, bufs, cache, sel, y)
        }

        /// Positions 0 .. CTX on the selected path, a dense twin in lockstep below top-k.
        fn run(&self) -> Result<RunOut, GateError> {
            let s = self.gpu.stream();
            let (mut a, mut b) = (self.mla(CTX)?, self.mla(CTX)?);
            let mut ix = self.ix(CTX, TOP_K)?;
            let mut out = RunOut {
                not_identical: Vec::new(),
                not_identity_list: Vec::new(),
                saved: Vec::new(),
                keys: Vec::new(),
                cache: Vec::new(),
                across: Vec::new(),
            };
            for p in 0..CTX {
                self.stage(&mut b, p)?;
                self.stage_ix(&mut ix, p, self.l.ik_row(p))?;
                self.selected(&mut b, &mut ix, None)?;
                if p < TOP_K {
                    self.stage(&mut a, p)?;
                    self.dense(&mut a)?;
                    s.synchronize()?;
                    if !bits_equal(&b.y.to_host_vec(s)?, &a.y.to_host_vec(s)?) {
                        out.not_identical.push(p);
                    }
                    if IDENTITY_LISTS.contains(&p) {
                        let list = ix.list.to_host_vec(s)?;
                        if !list[..=p].iter().enumerate().all(|(i, &v)| v as usize == i) {
                            out.not_identity_list.push(p);
                        }
                    }
                }
                if (ACROSS_AT..ACROSS_AT + ACROSS_M).contains(&p) {
                    s.synchronize()?;
                    out.across.push(b.y.to_host_vec(s)?);
                }
                if SAVES.contains(&p) {
                    s.synchronize()?;
                    let mut scores = ix.scratch.scores.to_host_vec(s)?;
                    scores.truncate(p + 1);
                    out.saved.push(Saved {
                        p,
                        y: b.y.to_host_vec(s)?,
                        list: ix.list.to_host_vec(s)?,
                        qt: ix.scratch.q.to_host_vec(s)?,
                        wt: ix.scratch.w.to_host_vec(s)?,
                        scores,
                    });
                }
            }
            s.synchronize()?;
            out.keys = ix.keys.buf().to_host_vec(s)?;
            out.cache = b.cache.buf().to_host_vec(s)?;
            Ok(out)
        }
    }

    // ------------------------------------------------------------------------- the clauses

    pub fn run() -> Result<(), GateError> {
        let gpu = Gpu::new()?;
        println!(
            "gate_kimi_k3_dsa: device {} — the lightning indexer at {IX_HEADS} heads of {HEAD_DIM}, \
             top-k {TOP_K} over {CTX} rows, the latent layer at {HEADS} heads, synthetic",
            gpu.device_name()?
        );
        let l = Layer::new(0x6b33_6473_6100_0001)?;
        let dev = Dev::new(&gpu, &l)?;
        let ks = keys(&l);
        let out = dev.run()?;
        let fault = gpu.fault()?;
        let mut results = Vec::new();
        results.push(clean("the run", fault));
        results.extend(key_clauses(&dev, &out)?);
        results.extend(list_clauses(&l, &out));
        results.extend(identity_clauses(&out));
        results.extend(selected_clauses(&l, &ks, &out));
        results.extend(fault_clauses(&dev)?);
        results.extend(refusal_clauses(&dev)?);
        results.extend(pass_clauses(&dev, &out)?);
        let failed = results.iter().filter(|p| !**p).count();
        println!(
            "gate_kimi_k3_dsa: {} — {} clauses, {failed} failed",
            verdict(failed == 0),
            results.len()
        );
        if failed == 0 {
            Ok(())
        } else {
            Err(checks_failed())
        }
    }

    fn clean(what: &str, fault: Option<bloomery_gpu::Fault>) -> bool {
        let pass = fault.is_none();
        println!(
            "  {what} raised no fault: {}: {}",
            fault.map_or_else(|| "none".to_string(), |f| f.to_string()),
            verdict(pass)
        );
        pass
    }

    fn key_clauses(dev: &Dev<'_>, out: &RunOut) -> Result<Vec<bool>, GateError> {
        let l = dev.l;
        let mut res = Vec::new();
        let off = (0..CTX)
            .filter(|&p| {
                out.keys[p * HEAD_DIM..(p + 1) * HEAD_DIM]
                    != key_host(l.ik_row(p), &l.ix_gain, &l.ix_bias)[..]
            })
            .count();
        let pass = off == 0;
        println!(
            "  key: the {CTX} rows the steps wrote vs the key kernel's host rule: {off} rows differ \
             in a bit: {}",
            verdict(pass)
        );
        res.push(pass);
        // A rows pass's launch: PASS_M keys at their positions of a fresh key store.
        let s = dev.gpu.stream();
        let mut keys = DeviceTensor::upload(s, &vec![KEY_FILL; CTX * HEAD_DIM], CTX, HEAD_DIM)?;
        let x =
            DeviceBuffer::from_host(s, &l.ik[PASS_AT * HEAD_DIM..(PASS_AT + PASS_M) * HEAD_DIM])?;
        let pos: Vec<u32> = (PASS_AT..PASS_AT + PASS_M)
            .map(u32::try_from)
            .collect::<Result<_, _>>()?;
        let pos = DeviceBuffer::from_host(s, &pos)?;
        enqueue_keys(
            s,
            &dev.ik,
            (&dev.ix_gain, &dev.ix_bias, IX_EPS),
            (&x, &pos, PASS_M),
            dev.gpu.unlabelled_sink(),
            &mut keys,
        )?;
        s.synchronize()?;
        let got = keys.buf().to_host_vec(s)?;
        let rows = PASS_AT * HEAD_DIM..(PASS_AT + PASS_M) * HEAD_DIM;
        let same = got[rows.clone()] == out.keys[rows.clone()];
        let beside = got[..rows.start]
            .iter()
            .chain(&got[rows.end..])
            .all(|&v| v == KEY_FILL);
        let pass = same && beside && dev.gpu.fault()?.is_none();
        println!(
            "  key: a rows pass of {PASS_M} keys from position {PASS_AT}: the steps' bits: {}, \
             every other row untouched: {}: {}",
            if same { "yes" } else { "no" },
            if beside { "yes" } else { "no" },
            verdict(pass)
        );
        res.push(pass);
        Ok(res)
    }

    fn list_clauses(l: &Layer, out: &RunOut) -> Vec<bool> {
        let mut res = Vec::new();
        for sv in &out.saved {
            let p = sv.p;
            let n = p + 1;
            let q_raw = &l.iq[p * IX_HEADS * HEAD_DIM..(p + 1) * IX_HEADS * HEAD_DIM];
            let qt: Vec<f32> = q_raw
                .chunks(HEAD_DIM)
                .flat_map(|h| {
                    let mut v = h.to_vec();
                    fast_ht(&mut v);
                    v
                })
                .collect();
            let wt: Vec<f32> = l.iw[p * IX_HEADS..(p + 1) * IX_HEADS]
                .iter()
                .map(|&w| w * (W_SCALE as f32))
                .collect();
            let inputs = bits_equal(&sv.qt, &qt) && bits_equal(&sv.wt, &wt);
            let top = exact_top(&sv.scores, TOP_K) == sv.list;
            let sc = scores(l, p, &sv.qt, &sv.wt, &out.keys[..n * HEAD_DIM]);
            let in_band = sv
                .scores
                .iter()
                .zip(sc.rule.iter().zip(&sc.kernel))
                .filter(|&(&c, (&r, &b))| outside((f64::from(c) - r).abs(), b))
                .count();
            let (off_rule, out_rule) = ids_vs(&sv.list, &sc.rule, &sc.kernel, TOP_K);
            let beta: Vec<f64> = sc
                .kernel
                .iter()
                .zip(sc.rule.iter().zip(&sc.graph))
                .map(|(&k, (&r, &g))| k + (r - g).abs())
                .collect();
            let (off_graph, out_graph) = ids_vs(&sv.list, &sc.graph, &beta, TOP_K);
            let pass = inputs && top && in_band == 0 && off_rule == 0 && off_graph == 0;
            println!(
                "  list at {n} rows: the score pass's query and weights the host's bits: {}, the \
                 list the exact top-{TOP_K} of the card's scores: {}, scores outside the kernel \
                 band of our rule: {in_band}, rows off our rule's top-k {off_rule} ({out_rule} in \
                 its tie band), off the graph's {off_graph} ({out_graph} in its tie band): {}",
                if inputs { "yes" } else { "no" },
                if top { "yes" } else { "no" },
                verdict(pass)
            );
            res.push(pass);
            if p == CTX - 1 {
                let floor = n / 100;
                for (what, refs) in [
                    ("without the head weights", &sc.no_weights),
                    ("without the relu", &sc.no_relu),
                ] {
                    let (off, _) = ids_vs(&sv.list, refs, &beta, TOP_K);
                    let pass = off > floor;
                    println!(
                        "  separation list at {n} rows: the card's list vs the graph {what}: {off} \
                         rows off, past {floor}: {}",
                        verdict(pass)
                    );
                    res.push(pass);
                }
            }
        }
        res
    }

    fn identity_clauses(out: &RunOut) -> Vec<bool> {
        let pass = out.not_identical.is_empty();
        println!(
            "  identity: positions 0 .. {TOP_K} selected vs dense, bit for bit: {} differ{}: {}",
            out.not_identical.len(),
            out.not_identical
                .first()
                .map_or_else(String::new, |p| format!(" (first at {p})")),
            verdict(pass)
        );
        let lists = out.not_identity_list.is_empty();
        println!(
            "  identity: the list is 0 ..= pos at positions {IDENTITY_LISTS:?}: not at {:?}: {}",
            out.not_identity_list,
            verdict(lists)
        );
        vec![pass, lists]
    }

    fn selected_clauses(l: &Layer, ks: &Keys, out: &RunOut) -> Vec<bool> {
        let mut res = Vec::new();
        for sv in &out.saved {
            let p = sv.p;
            let rows: Vec<usize> = sv.list.iter().map(|&i| i as usize).collect();
            let err = rel_l2(&sv.y, &graph(l, ks, p, &rows));
            let pass = err <= BAND;
            println!(
                "  selected at {} keys: the gated heads vs the graph over the card's {TOP_K} rows: \
                 rel-L2 {err:.3e} (band {BAND:.0e}): {}",
                p + 1,
                verdict(pass)
            );
            res.push(pass);
            if p == CTX - 1 {
                let all: Vec<usize> = (0..=p).collect();
                let first: Vec<usize> = (0..TOP_K).collect();
                for (what, rows) in [("every key", all), ("the first top-k rows", first)] {
                    let sep = rel_l2(&sv.y, &graph(l, ks, p, &rows));
                    let pass = sep > SEP * BAND;
                    println!(
                        "  separation selected at {} keys: the card vs the graph over {what}: \
                         rel-L2 {sep:.3e}, past {SEP}x band: {}",
                        p + 1,
                        verdict(pass)
                    );
                    res.push(pass);
                }
            }
        }
        res
    }

    /// Whether `fault` names the launch's layer and `site`; clears the word after.
    fn raised(gpu: &Gpu, what: &str, site: FaultSite) -> Result<bool, GateError> {
        let f = gpu.fault()?;
        let pass = f.is_some_and(|f| f.layer == LAYER_NONE && f.sites & (1 << site as u32) != 0);
        println!(
            "  fault {what}: {}: {}",
            f.map_or_else(|| "nothing raised".to_string(), |f| f.to_string()),
            verdict(pass)
        );
        gpu.clear_fault()?;
        Ok(pass)
    }

    fn fault_clauses(dev: &Dev<'_>) -> Result<Vec<bool>, GateError> {
        let gpu = dev.gpu;
        let s = gpu.stream();
        gpu.clear_fault()?;
        let mut res = Vec::new();
        // A list entry at the cache's height.
        let mut m = dev.mla(F_CTX)?;
        let mut ix = dev.ix(F_CTX, F_TOP_K)?;
        dev.stage(&mut m, 0)?;
        dev.stage_ix(&mut ix, 0, dev.l.ik_row(0))?;
        let mut poison = vec![0u32; F_TOP_K];
        poison[0] = u32::try_from(F_CTX)?;
        dev.selected(&mut m, &mut ix, Some(&poison))?;
        res.push(raised(
            gpu,
            &format!("a list entry {F_CTX} over a cache of {F_CTX} rows (the gather)"),
            FaultSite::CachePos,
        )?);
        // A key projection holding a NaN.
        let mut nan = dev.l.ik_row(1).to_vec();
        nan[5] = f32::NAN;
        dev.stage(&mut m, 1)?;
        dev.stage_ix(&mut ix, 1, &nan)?;
        dev.selected(&mut m, &mut ix, None)?;
        res.push(raised(
            gpu,
            "a key projection holding a NaN (the key launch)",
            FaultSite::CacheValue,
        )?);
        // A key position at the key rows' height.
        let pos = DeviceBuffer::from_host(s, &[u32::try_from(F_CTX)?])?;
        enqueue_keys(
            s,
            &dev.ik,
            (&dev.ix_gain, &dev.ix_bias, IX_EPS),
            (&ix.ik, &pos, 1),
            gpu.unlabelled_sink(),
            &mut ix.keys,
        )?;
        res.push(raised(
            gpu,
            &format!("a key position {F_CTX} over {F_CTX} key rows (the key launch)"),
            FaultSite::CachePos,
        )?);
        Ok(res)
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

    fn sel<'b>(
        list: &'b DeviceBuffer<u32>,
        n: &'b DeviceBuffer<u32>,
        stride: usize,
        scratch: &'b mut DeviceTensor<u16>,
    ) -> Selection<'b> {
        Selection {
            list,
            n,
            stride,
            scratch,
        }
    }

    fn refusal_clauses(dev: &Dev<'_>) -> Result<Vec<bool>, GateError> {
        let s = dev.gpu.stream();
        let ixr = Indexer {
            heads: IX_HEADS,
            d: HEAD_DIM,
            top_k: TOP_K,
            norm_eps: IX_EPS,
            layers: vec![false, false, true, false],
        };
        let mut res = vec![
            refusal(
                "a selection on a file with no indexer",
                refused(check_selection(None, TOP_K)),
                "no indexer",
            ),
            refusal(
                "a list of 64 rows under top-k 2048",
                refused(check_selection(Some(&ixr), 64)),
                "top_k picks 2048",
            ),
            refusal(
                "a step at position u32::MAX",
                refused(count_words(u32::MAX, 2048)),
                "u32's last value",
            ),
        ];
        // `attend_sel` with one selection part of a wrong shape and the rest a 64-row list over a
        // 128-row cache.
        let mut m = dev.mla(F_CTX)?;
        dev.stage(&mut m, 0)?;
        let list = DeviceBuffer::from_host(s, &vec![0u32; F_TOP_K])?;
        let short = DeviceBuffer::from_host(s, &[0u32; F_TOP_K / 2])?;
        let one = DeviceBuffer::from_host(s, &[1u32])?;
        let three = DeviceBuffer::from_host(s, &[1u32; 3])?;
        let mut narrow = DeviceTensor::<u16>::zeroed(s, F_TOP_K, ROW - 32)?;
        let mut low = DeviceTensor::<u16>::zeroed(s, F_TOP_K / 2, ROW)?;
        let mut good = DeviceTensor::<u16>::zeroed(s, F_TOP_K, ROW)?;
        let mut tall = DeviceTensor::<u16>::zeroed(s, F_CTX, ROW)?;
        let tall_list = DeviceBuffer::from_host(s, &vec![0u32; F_CTX])?;
        res.push(refusal(
            "a scratch 544 wide",
            refused(dev.core(&mut m, sel(&list, &one, F_TOP_K, &mut narrow))),
            "64x544",
        ));
        res.push(refusal(
            "a scratch of 32 rows under a list of 64",
            refused(dev.core(&mut m, sel(&list, &one, F_TOP_K, &mut low))),
            "32x576",
        ));
        res.push(refusal(
            "a list of 32 entries under a width of 64",
            refused(dev.core(&mut m, sel(&short, &one, F_TOP_K, &mut good))),
            "32 entries",
        ));
        res.push(refusal(
            "three count words",
            refused(dev.core(&mut m, sel(&list, &three, F_TOP_K, &mut good))),
            "3 count words",
        ));
        res.push(refusal(
            "a list as tall as the cache",
            refused(dev.core(&mut m, sel(&tall_list, &one, F_CTX, &mut tall))),
            "fewer rows than the cache holds",
        ));
        Ok(res)
    }

    /// The pass clause's buffers: the rows' inputs, one row's attention buffers, the pass's cache
    /// and key rows, and one row's score scratch, list and gathered rows.
    struct Across {
        kv: DeviceBuffer<f32>,
        q: DeviceBuffer<f32>,
        gate: DeviceBuffer<f32>,
        pos: DeviceBuffer<u32>,
        n_keys: DeviceBuffer<u32>,
        ik: DeviceBuffer<f32>,
        iq: DeviceBuffer<f32>,
        iw: DeviceBuffer<f32>,
        cache: DeviceTensor<u16>,
        keys: DeviceTensor<u16>,
        one: Mla,
        scratch: IndexerScratch,
        list: DeviceBuffer<u32>,
        gathered: DeviceTensor<u16>,
        y: DeviceBuffer<f32>,
    }

    impl Dev<'_> {
        /// The pass clause's buffers: the steps' rows below [`ACROSS_AT`] and [`KEY_FILL`] from
        /// it in the cache and the key rows, the seeded inputs of the pass's rows; and its count
        /// words from [`pass_words`].
        fn across(&self, out: &RunOut) -> Result<(Across, DeviceBuffer<u32>), GateError> {
            let s = self.gpu.stream();
            let l = self.l;
            let (a, e) = (ACROSS_AT, ACROSS_AT + ACROSS_M);
            let hv = HEADS * V_HEAD;
            let below = |rows: &[u16], width: usize| -> Vec<u16> {
                let mut v = rows[..a * width].to_vec();
                v.resize(CTX * width, KEY_FILL);
                v
            };
            let pos: Vec<u32> = (a..e).map(u32::try_from).collect::<Result<_, _>>()?;
            let n_keys: Vec<u32> = pos.iter().map(|p| p + 1).collect();
            let mut words = vec![0u32; ACROSS_M * WORDS];
            pass_words(&mut words, u32::try_from(a)?, u32::try_from(TOP_K)?)?;
            let f = |v: &[f32]| DeviceBuffer::from_host(s, v);
            let x = Across {
                kv: f(&l.kv[a * ROW..e * ROW])?,
                q: f(&l.q[a * HEADS * HEAD_K..e * HEADS * HEAD_K])?,
                gate: f(&l.gate[a * hv..e * hv])?,
                pos: DeviceBuffer::from_host(s, &pos)?,
                n_keys: DeviceBuffer::from_host(s, &n_keys)?,
                ik: f(&l.ik[a * HEAD_DIM..e * HEAD_DIM])?,
                iq: f(&l.iq[a * IX_HEADS * HEAD_DIM..e * IX_HEADS * HEAD_DIM])?,
                iw: f(&l.iw[a * IX_HEADS..e * IX_HEADS])?,
                cache: DeviceTensor::upload(s, &below(&out.cache, ROW), CTX, ROW)?,
                keys: DeviceTensor::upload(s, &below(&out.keys, HEAD_DIM), CTX, HEAD_DIM)?,
                one: self.mla(CTX)?,
                scratch: IndexerScratch::new(s, 1, CTX)?,
                list: DeviceBuffer::zeroed(s, TOP_K)?,
                gathered: DeviceTensor::zeroed(s, TOP_K, ROW)?,
                y: DeviceBuffer::zeroed(s, ACROSS_M * hv)?,
            };
            Ok((x, DeviceBuffer::from_host(s, &words)?))
        }

        /// [`attend_rows_sel`] over `x`'s `m` rows, its count words `words`.
        fn attend_across(
            &self,
            x: &mut Across,
            m: usize,
            words: &DeviceBuffer<u32>,
        ) -> Result<(), GpuError> {
            let Across {
                kv,
                q,
                gate,
                pos,
                n_keys,
                iq,
                iw,
                cache,
                keys,
                one,
                scratch,
                list,
                gathered,
                y,
                ..
            } = x;
            attend_rows_sel(
                self.gpu,
                (&self.mk, &self.ik),
                &self.d,
                m,
                AttendIn {
                    kv,
                    q,
                    gate,
                    kv_norm: &self.gain,
                    k_b: (&self.kb.0, &self.kb.1),
                    v_b: ValueW::Q8_0(&self.vb.0, &self.vb.1),
                    pos,
                    n_keys,
                    fault: self.gpu.unlabelled_sink(),
                },
                AttnBufs {
                    qabs: &mut one.qabs,
                    att: &mut one.att,
                    av: &mut one.av,
                    part_v: &mut one.part_v,
                    part_ms: &mut one.part_ms,
                },
                cache,
                PassPick {
                    keys,
                    iq,
                    iw,
                    words,
                    scratch,
                    list,
                    gathered,
                },
                y,
            )
        }
    }

    fn pass_clauses(dev: &Dev<'_>, out: &RunOut) -> Result<Vec<bool>, GateError> {
        let s = dev.gpu.stream();
        let (mut x, words) = dev.across(out)?;
        enqueue_keys(
            s,
            &dev.ik,
            (&dev.ix_gain, &dev.ix_bias, IX_EPS),
            (&x.ik, &x.pos, ACROSS_M),
            dev.gpu.unlabelled_sink(),
            &mut x.keys,
        )?;
        dev.attend_across(&mut x, ACROSS_M, &words)?;
        s.synchronize()?;
        let fault = dev.gpu.fault()?;
        let hv = HEADS * V_HEAD;
        let y = x.y.to_host_vec(s)?;
        let heads_off: Vec<usize> = (0..ACROSS_M)
            .filter(|&t| {
                out.across
                    .get(t)
                    .is_none_or(|want| !bits_equal(&y[t * hv..(t + 1) * hv], want))
            })
            .map(|t| ACROSS_AT + t)
            .collect();
        let (a, e) = (ACROSS_AT, ACROSS_AT + ACROSS_M);
        let keys = x.keys.buf().to_host_vec(s)?;
        let cache = x.cache.buf().to_host_vec(s)?;
        let keys_same = keys[a * HEAD_DIM..e * HEAD_DIM] == out.keys[a * HEAD_DIM..e * HEAD_DIM];
        let cache_same = cache[a * ROW..e * ROW] == out.cache[a * ROW..e * ROW];
        let pass = fault.is_none() && heads_off.is_empty() && keys_same && cache_same;
        println!(
            "  pass: {ACROSS_M} rows from position {ACROSS_AT} across top-k {TOP_K} through \
             attend_rows_sel vs the steps: rows whose gated heads differ in a bit {heads_off:?}, \
             key rows the steps' bits: {}, cache rows the steps' bits: {}, fault {}: {}",
            if keys_same { "yes" } else { "no" },
            if cache_same { "yes" } else { "no" },
            fault.map_or_else(|| "none".to_string(), |f| f.to_string()),
            verdict(pass)
        );
        let short = Window::<u32>::of(&words, 0, 2 * WORDS)?;
        let refusal_short = refusal(
            "a pass of 8 rows over the count words of 2",
            refused(dev.attend_across(&mut x, ACROSS_M, &short)),
            "words of 6",
        );
        Ok(vec![pass, refusal_short])
    }
}
