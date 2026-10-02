//! GPU gate for DeepSeek-V4.1's lightning indexer — B4 op block F
//! (`docs/research/v41-b4-plan-report.md` §1-F) — `bloomery_gpu_deepseek41::
//! indexer` against ik's CPU dumps. Every emitting layer
//! (`LayerKind::indexer`) of the decode-step sets at position 301 (top_k 64,
//! the set's `--override-kv`) and 1,025 (the file's top_k), each fused (ik's
//! bucket top-k, `lid_top_k`) and unfused (`lid_kq`, `lid_score`,
//! `lid_score_masked` and an argsort), gets two launches of the op:
//!
//! - run A, the engine's path: our gemv of the dump's `qr_norm-L` in the
//!   file's type of `attn_q_b` (q8_0 on f32 activations, or q3_K on q8_1
//!   activations) and our q3_K gemv of its `attn_norm-L` (q8_1 activations)
//!   feed the indexer;
//! - run B, ik's projections: the dump's `lid_q-L` and `lid_weights-L` feed
//!   it, so everything past the gemvs meets ik's own inputs.
//!
//! Per layer, the B4 gate form:
//! - (i) the query after rope and Hadamard, and the scaled weights. Run A's
//!   against this binary's transcription of our rope and transform on its own
//!   gemv output and of the scale, bit for bit, and against the dump
//!   (`lid_q_hadamard`, the `SCALE` node) within a band derived from each
//!   gemv's distance to ik's ([`q8_rows`], `act_rule::q3k_rows`) carried
//!   through the rope and the transform ([`query_band`]); a q3_K query gemv
//!   also by itself, each row against the exact dot of our q8_1 values
//!   within `KERNEL_BAND` and against `lid_q` within its row's band. Run B's
//!   bit for bit against the dump: our table, pairs and transform are ik's
//!   op for op.
//!   ik's rule is simulated as well: its dot against `lid_q` (q8_2 under a
//!   q8_0 `attn_q_b`, `act_rule::dot_q3k` on q8_K under a q3_K one) and its
//!   rope and transform against `indexer_q` and `lid_q_hadamard` bit for bit,
//!   its q3_K dot of `proj` within its lanes' roundings of `lid_weights`.
//! - (ii) the scores of both runs against our rule ([`rule`]) within the
//!   tensor-core band; run B's against the unfused dump's `lid_score` within
//!   that band, our split's representation error and ik's own band.
//! - (iii) the ids: run B's list and ik's `lid_top_k`, fused and unfused,
//!   each against the rule's selection at every row outside the tie band of
//!   the k-th row ([`ids_vs_rule`]); each run's list is exactly the top-k of
//!   the kernel's own scores (order-preserving keys, ties to the lower row),
//!   `k` entries in ascending order with the rest of the row untouched; the
//!   unfused mask hides exactly the rows from `n_vis` on.
//! - (iv) the identity: at `…_d1n_every_node` (the file's top_k over at most
//!   302 rows, where ik builds no indexer) the list is `0..n_vis` and the
//!   score pass writes nothing; the selected-row attention over such a list
//!   is bit-identical to the prefix attention.
//! - (v) depth cases the sets cannot reach: synthetic keys, query and weights
//!   at 16,384 and 32,768 visible rows of a longer cache, with a group of
//!   duplicated key rows planted to straddle the threshold (bit-identical
//!   scores, so the lower rows must win), against our rule and its
//!   selection; and a clustered case whose scores share their top bits,
//!   against the exact top-k of the kernel's scores.
//! - (vi) reruns bit-identical; one captured graph replayed over rewritten
//!   counts (`n_vis` and `top_k`, selecting and identity), each replay
//!   bit-identical to an eager run; no local depot in either entry.
//! - (vii) the candidate mask against ik's (the `cand-deepseek41` set
//!   `…_d1c_unfused_every_node`: position 301, top_k 64, 16 kept blocks of 8, ik's
//!   separate V4.1 graph). ik's graph applies no Hadamard transform to the
//!   indexer's query and keys, ours to both; the transform is orthonormal, so
//!   the gate turns ik's keys into our basis (`fast_ht`, then f16) and holds
//!   the scores to ik's exact rule within the transform's and the f16's
//!   roundings ([`Basis`]) besides the kernel's and ik's own bands. Three
//!   checks, each with its tie band counted and printed:
//!   1. the rule against ik: `ds41_cand::kept_rule` over ik's
//!      `cand_block_score-20` (the pool-max of its masked scores; the blocks
//!      ik pads past the visible rows hold the pool's seed, `−f32::MAX`) gives ik's `cand_block_top_k-20`
//!      as a set and `cand_keep-20`'s 0 / −inf pattern, outside equal keys at
//!      the last kept place;
//!   2. our kernels on ik's inputs: layer 20's score pass and
//!      `CandKernels::enqueue_select` keep ik's blocks outside the band of the
//!      last kept block's key, and exactly the rule's blocks of our own
//!      scores;
//!   3. the consumers (24, 28, 32, 36): score, compaction over our kept
//!      blocks, the top-k over the counts view and the remap give the
//!      rule's list exactly (`ds41_cand::check`), and ik's `lid_top_k`
//!      outside the tie band of the k-th candidate row; ik's own mask hides
//!      exactly the rows outside its kept blocks. Red when no consumer's list
//!      differs from the unmasked top-k of its scores: such data could not
//!      see a consumer that ignores the mask.
//!
//! Every output is poisoned before a launch (NaN query, weights and scores,
//! `u32::MAX` list entries), so a slot written where it should not be, or
//! not written, shows, and the histogram must be zero after every pair.

#[cfg(not(feature = "deepseek41"))]
fn main() {
    eprintln!(
        "gate_deepseek41_index: built without the `deepseek41` feature; see `just gate-gpu-ds41-index`."
    );
    std::process::exit(2);
}

#[cfg(feature = "deepseek41")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_deepseek41_index", gate::run())
}

#[cfg(feature = "deepseek41")]
#[path = "shared/ds41_cand.rs"]
mod ds41_cand;

#[cfg(feature = "deepseek41")]
mod gate {
    use std::collections::HashMap;
    use std::f32::consts::FRAC_1_SQRT_2;

    use bloomery_gpu::cand::{
        CandKernels, CandScratch, CandShape, CompactArgs, RemapArgs, SelectArgs,
    };
    use bloomery_gpu::weights::{DevWeight, q8_0_planes, upload_file_tensor};
    use bloomery_gpu::{DeviceTensor, Gpu, GpuError, Q8Act};
    use bloomery_gpu_deepseek41::attn::{self, AttnArgs, AttnKernels, LATENT, SelectedRows};
    use bloomery_gpu_deepseek41::index_key::HT_SCALE;
    use bloomery_gpu_deepseek41::indexer::{
        HEAD_DIM, HEADS, IndexerArgs, IndexerKernels, IndexerScratch,
    };
    use bloomery_gpu_deepseek41::params::rope_specs;
    use bloomery_gpu_deepseek41::rope::{Direction, RopeSpec, RopeTable, ggml_rope_cache};
    use bloomery_gpu_gates::act_rule::{self, Q3kWeight};
    use bloomery_gpu_gates::ik_q8_2::{self, QK, folded, half_sum};
    use bloomery_gpu_gates::oracle::deepseek41::{D1, D1_UNFUSED, D1N, D2, D2_UNFUSED};
    use bloomery_gpu_gates::oracle::for_arch;
    use bloomery_gpu_gates::{
        GateError, KERNEL_BAND, Layout, NAN_F16, RefManifest, RefRow, RowKind, activations,
        bits_equal, checks_failed, max_rel_err, no_local_depot, ref_ints, ref_model_path,
        ref_tensor_logical_in, verdict, widened_f16_rows_in,
    };
    use cuda_core::{CudaStream, DeviceBuffer};
    use gguf::Split;
    use gguf::quant::{GgmlType, Q8Block, f32_to_f16_bits, half_to_f32};
    use model::arch::Arch;
    use model::arch::deepseek41::hparams::{CandidateRole, Hparams};
    use model::arch::deepseek41::plan::{Planner, StepPlan};
    use refset::arch::deepseek41::{CAND, D1C};

    use super::ds41_cand;

    /// f32's unit roundoff, 2⁻²⁴.
    const U: f64 = f32::EPSILON as f64 / 2.0;
    /// The tensor-core dot's bound in units of `U` times its products'
    /// magnitude [derived]: an `mma` forms every f16 product exactly and adds
    /// its sixteen products and the accumulator with each addend cut at the
    /// 24th bit of the largest and the sum rounded or cut once more, so one
    /// k-step errs by at most 18·2⁻²³ = 36u of its addends' magnitude; the
    /// accumulator a step adds carries the magnitude of the steps before it,
    /// so the eight k-steps of a head err by at most 8·36u = 288u of the
    /// head's product magnitude.
    const TC_UNITS: f64 = 288.0;
    /// Roundings one head's weighted term meets in the kernel's head sum: a
    /// product and three fused multiply-adds along its lane group, then three
    /// butterfly levels.
    const HEAD_SUM_ROUNDINGS: usize = 7;
    /// What the gate fills the float outputs with before a launch.
    const POISON: f32 = f32::NAN;
    /// What it fills the lists with.
    const LIST_POISON: u32 = u32::MAX;
    /// List entries past top_k in every token's row: they must stay poisoned.
    const LIST_SLACK: usize = 16;
    /// Where the counts and the table start in the gate's buffers: not at
    /// zero, so an offset the kernels drop shows.
    const N_VIS_AT: usize = 3;
    const ROPE_AT: usize = 5;
    /// Host threads for the rules.
    const HOST_THREADS: usize = 16;
    /// Rows of the planted tie group: the threshold row and its copies.
    const GROUP: usize = 5;

    /// `γ(n) = n·u / (1 − n·u)`: the relative bound on a term that went
    /// through `n` roundings of its partial sums.
    fn gamma(n: usize) -> f64 {
        let nu = n as f64 * U;
        nu / (1.0 - nu)
    }

    /// A distance past its band; a NaN distance is past every band.
    fn outside(d: f64, band: f64) -> bool {
        d.is_nan() || d > band
    }

    fn list(v: &[impl ToString]) -> String {
        v.iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(",")
    }

    /// The top-k override in a set's `# flags` — ik's `--override-kv
    /// <arch>.attention.indexer.top_k=int:<n>` — if there is one.
    fn top_k_override(flags: &str) -> Result<Option<usize>, GateError> {
        kv_override(flags, "attention.indexer.top_k")
    }

    /// The integer override of key `<arch>.<suffix>` in a set's `# flags`
    /// (`--override-kv <arch>.<suffix>=int:<n>`), if there is one.
    fn kv_override(flags: &str, suffix: &str) -> Result<Option<usize>, GateError> {
        let key = format!("{}.{suffix}=int:", Arch::Deepseek41.name());
        let mut it = flags.split_whitespace();
        let mut found = None;
        while let Some(tok) = it.next() {
            if tok == "--override-kv" {
                let kv = it.next().ok_or("# flags: --override-kv without a value")?;
                if let Some(v) = kv.strip_prefix(key.as_str()) {
                    found = Some(v.parse::<usize>()?);
                }
            }
        }
        Ok(found)
    }

    // ------------------------------------------------------------ weights

    /// A q8_0 weight: its blocks for the host rules, its q8f32 planes for the
    /// gemv (the loader's own `q8_0_planes`).
    struct Q8 {
        blocks: Vec<Q8Block>,
        k: usize,
        rows: usize,
        qs: DeviceTensor<u32>,
        d: DeviceTensor<u16>,
    }

    fn load_q8(split: &Split, stream: &CudaStream, name: &str) -> Result<Q8, GateError> {
        let (s, t) = split
            .find(name)
            .ok_or_else(|| format!("{name} is not in the model file"))?;
        let [k, rows] = match t.dims.as_slice() {
            &[k, rows] if t.ty == GgmlType::Q8_0 && k.is_multiple_of(QK as u64) => {
                [usize::try_from(k)?, usize::try_from(rows)?]
            }
            _ => return Err(format!("{name} is {:?} {:?}, want a 2-D Q8_0", t.ty, t.dims).into()),
        };
        let bytes = split.shard(s).ok_or("shard index out of range")?.data(t)?;
        let blocks: Vec<Q8Block> = bytes
            .as_chunks::<34>()
            .0
            .iter()
            .map(Q8Block::from_bytes)
            .collect();
        if blocks.len() != rows * k / QK {
            return Err(format!("{name}: {} blocks, want {}", blocks.len(), rows * k / QK).into());
        }
        let (qs, d) = q8_0_planes(&blocks);
        Ok(Q8 {
            qs: DeviceTensor::upload(stream, &qs, rows, k / 4)?,
            d: DeviceTensor::upload(stream, &d, rows, k / QK)?,
            blocks,
            k,
            rows,
        })
    }

    /// A q3_K weight on the host as both rules read it, and on the device
    /// in the loader's own packing (`upload_file_tensor`).
    struct Q3 {
        host: Q3kWeight,
        dev: DeviceTensor<u32>,
    }

    fn load_q3(split: &Split, stream: &CudaStream, name: &str) -> Result<Q3, GateError> {
        let (s, t) = split
            .find(name)
            .ok_or_else(|| format!("{name} is not in the model file"))?;
        let &[k, rows] = t.dims.as_slice() else {
            return Err(format!("{name} has dims {:?}, want [K, rows]", t.dims).into());
        };
        let (k, rows) = (usize::try_from(k)?, usize::try_from(rows)?);
        if t.ty != GgmlType::Q3_K {
            return Err(format!("{name} is {:?} K={k}, want q3_K", t.ty).into());
        }
        let shard = split.shard(s).ok_or("shard index out of range")?;
        let host = Q3kWeight::new(shard.data(t)?, k, rows).map_err(|e| format!("{name}: {e}"))?;
        let DevWeight::KQuant { w: dev, .. } = upload_file_tensor(stream, shard, t)? else {
            return Err(format!("{name}: the loader did not pack it as a K-quant").into());
        };
        Ok(Q3 { host, dev })
    }

    // ----------------------------------------------------------- gemv bands

    /// One q8_0 row: ik's value (`ik_q8_2::dot` on its q8_2 activations) and
    /// the bound on our gemv's distance to it. Per value v ik multiplies the
    /// weight `w_v` by its reconstruction `x̂_v = sign(w_v)·folded_v·d_x` of
    /// the activation where ours multiplies by `x_v`, so the exact sums differ
    /// by at most `Σ|w_v|·|x_v − x̂_v|`; our f32 accumulation adds at most
    /// `γ(k/32 + 5)` of `Σ|w_v x_v|` (k/32 multiply-adds along a lane, five
    /// butterfly levels) and ik's `γ(k/128 + 3)` of `Σ|d_w d_x p|` over its
    /// partials (k/128 along a lane, three levels of `hsum_float_8`).
    #[derive(Clone, Copy, Default)]
    struct Q8Row {
        ik: f32,
        band: f64,
    }

    fn q8_row(blocks: &[Q8Block], x: &[f32], xq: &[i8], xd: &[f32]) -> Q8Row {
        let k = x.len();
        let (mut abs_ours, mut abs_ik, mut act) = (0.0f64, 0.0f64, 0.0f64);
        for (b, blk) in blocks.iter().enumerate() {
            let (dw, dx) = (half_to_f32(blk.d), xd[b]);
            let a = &xq[b * QK..(b + 1) * QK];
            for h in 0..2 {
                let t = f64::from(dw * dx) * f64::from(half_sum(blk, a, h));
                abs_ik += t.abs();
            }
            for (j, &c) in blk.q.iter().enumerate() {
                let wv = f64::from(f32::from(c) * dw);
                let xv = f64::from(x[b * QK + j]);
                abs_ours += (wv * xv).abs();
                let xh = f64::from(c.signum()) * f64::from(folded(c, a[j])) * f64::from(dx);
                act += wv.abs() * (xv - xh).abs();
            }
        }
        Q8Row {
            ik: ik_q8_2::dot(blocks, xq, xd),
            band: act + gamma(k / QK + 5) * abs_ours + gamma(k / (4 * QK) + 3) * abs_ik,
        }
    }

    /// [`q8_row`] for every row of `w` against one column `x`.
    fn q8_rows(w: &Q8, x: &[f32]) -> Vec<Q8Row> {
        let nb = w.k / QK;
        let (xq, xd) = ik_q8_2::quantize(x);
        let (xq, xd) = (&xq[..], &xd[..]);
        let mut out = vec![Q8Row::default(); w.rows];
        let chunk = w.rows.div_ceil(HOST_THREADS);
        std::thread::scope(|s| {
            for (c, part) in out.chunks_mut(chunk).enumerate() {
                s.spawn(move || {
                    for (i, o) in part.iter_mut().enumerate() {
                        let r = c * chunk + i;
                        *o = q8_row(&w.blocks[r * nb..(r + 1) * nb], x, xq, xd);
                    }
                });
            }
        });
        out
    }

    // ------------------------------------------------------ query rules

    /// The tail rope of one head: its last `cs.len()` values turned in pairs
    /// by `cs` (`[cos, sin, …]`), every product and sum rounded on its own —
    /// `rope::rope_pair_rn` and ik's rotation alike.
    fn rotate(x: &[f32], cs: &[f32]) -> Vec<f32> {
        let mut y = x.to_vec();
        let tail = y.len() - cs.len();
        for (p, &[c, s]) in y[tail..]
            .as_chunks_mut::<2>()
            .0
            .iter_mut()
            .zip(cs.as_chunks::<2>().0)
        {
            let [x0, x1] = *p;
            *p = [x0 * c - x1 * s, x0 * s + x1 * c];
        }
        y
    }

    /// ik's `fast_ht` (`iqk_cpu_ops.cpp`): butterflies at `h = 1, 2, 4, …`,
    /// `x + y` and `x − y`, the scale multiplied by `0.707106781f` (the f32
    /// `FRAC_1_SQRT_2`) per stage, then every value by it.
    fn fast_ht(v: &mut [f32]) -> f32 {
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
            scale *= FRAC_1_SQRT_2;
            h <<= 1;
        }
        for x in v.iter_mut() {
            *x *= scale;
        }
        scale
    }

    /// Our query of one token (and ik's, whose ops are the same): per head
    /// the tail rope by `cs`, then the transform.
    fn query(q_raw: &[f32], cs: &[f32]) -> Vec<f32> {
        q_raw
            .chunks(HEAD_DIM)
            .flat_map(|h| {
                let mut v = rotate(h, cs);
                fast_ht(&mut v);
                v
            })
            .collect()
    }

    /// The bound on our query's distance to ik's, one value per head, from
    /// the gemv rows' bands `dl`: the rope turns pair `(x0, x1)` into
    /// `(x0·c − x1·s, x0·s + x1·c)`, so the pair's deltas mix by `|c|`,
    /// `|s|`, and each side rounds each output twice (`γ(2)` of its terms);
    /// the transform adds or subtracts every input into every output and
    /// scales by `S`, so an output moves by at most `S·Σ` of the inputs'
    /// deltas, and each side's seven butterflies and the scale round an
    /// output at most eight times (`γ(8)` of `S·Σ|x|`). Magnitudes are ours
    /// plus the delta, which bounds ik's.
    fn query_band(q_raw: &[f32], dl: &[f64], cs: &[f32]) -> Vec<f64> {
        let s = f64::from(HT_SCALE);
        let tail = HEAD_DIM - cs.len();
        (0..HEADS)
            .map(|h| {
                let x = &q_raw[h * HEAD_DIM..(h + 1) * HEAD_DIM];
                let dd = &dl[h * HEAD_DIM..(h + 1) * HEAD_DIM];
                let (mut dsum, mut xsum) = (0.0f64, 0.0f64);
                for d in 0..tail {
                    dsum += dd[d];
                    xsum += f64::from(x[d].abs()) + dd[d];
                }
                for (i, &[c, sn]) in cs.as_chunks::<2>().0.iter().enumerate() {
                    let (d0, d1) = (tail + 2 * i, tail + 2 * i + 1);
                    let (a0, a1) = (
                        f64::from(x[d0].abs()) + dd[d0],
                        f64::from(x[d1].abs()) + dd[d1],
                    );
                    let (ac, asn) = (f64::from(c.abs()), f64::from(sn.abs()));
                    let (m0, m1) = (a0 * ac + a1 * asn, a0 * asn + a1 * ac);
                    dsum += ac * dd[d0] + asn * dd[d1] + 2.0 * gamma(2) * m0;
                    dsum += asn * dd[d0] + ac * dd[d1] + 2.0 * gamma(2) * m1;
                    xsum += m0 + m1;
                }
                s * dsum + 2.0 * gamma(8) * s * xsum
            })
            .collect()
    }

    // ------------------------------------------------------ score rules

    /// The split of one query value the kernel makes: `hi = f16(q)`,
    /// `lo = f16(q − hi)`, each widened back.
    fn split(q: f32) -> (f32, f32) {
        let hi = half_to_f32(f32_to_f16_bits(q));
        let lo = half_to_f32(f32_to_f16_bits(q - hi));
        (hi, lo)
    }

    /// Our rule over one token's rows, and three bands per row.
    struct Rule {
        /// `Σ_h w_h · relu(Σ_d (hi + lo)·k)`: the kernel's split query, its
        /// weights, the exact f16 products summed in f64.
        score: Vec<f64>,
        /// The kernel's distance to `score`: per head the tensor cores'
        /// [`TC_UNITS`] of the products' magnitude and one rounding of
        /// `acc_hi + acc_lo`, carried by `|w_h|` (relu moves nothing further);
        /// then the head sum's [`HEAD_SUM_ROUNDINGS`].
        kernel: Vec<f64>,
        /// The split's distance to the f32 query: `Σ_h |w_h| Σ_d |q − hi −
        /// lo|·|k|`.
        rep: Vec<f64>,
        /// ik's distance to the exact score of the f32 query and the same
        /// weights: a 128-long f32 dot per head (`γ(128)` of `Σ|q·k|`), its
        /// head sum (`γ(32)` of `Σ|w·r|`, the fused path's fused
        /// multiply-adds; the unfused path's product and f64 sum sit inside
        /// it), one final rounding.
        ik: Vec<f64>,
    }

    fn rule(q: &[f32], w: &[f32], keys: &[u16], n: usize) -> Rule {
        let sp: Vec<[f64; 3]> = q
            .iter()
            .map(|&v| {
                let (hi, lo) = split(v);
                [f64::from(hi), f64::from(lo), f64::from(v)]
            })
            .collect();
        let w64: Vec<f64> = w.iter().map(|&v| f64::from(v)).collect();
        let mut rows = vec![[0.0f64; 4]; n];
        let chunk = n.div_ceil(HOST_THREADS).max(1);
        let (sp, w64) = (&sp, &w64);
        std::thread::scope(|s| {
            for (c, part) in rows.chunks_mut(chunk).enumerate() {
                s.spawn(move || {
                    for (i, o) in part.iter_mut().enumerate() {
                        let t = c * chunk + i;
                        let k: Vec<f64> = keys[t * HEAD_DIM..(t + 1) * HEAD_DIM]
                            .iter()
                            .map(|&b| f64::from(half_to_f32(b)))
                            .collect();
                        let (mut sc, mut kb, mut rep, mut ikb, mut smag) =
                            (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
                        for (h, &wh) in w64.iter().enumerate() {
                            let (mut dot, mut p, mut pq, mut r) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
                            for (&[hi, lo, qv], &kv) in
                                sp[h * HEAD_DIM..(h + 1) * HEAD_DIM].iter().zip(&k)
                            {
                                dot += (hi + lo) * kv;
                                p += (hi * kv).abs() + (lo * kv).abs();
                                pq += (qv * kv).abs();
                                r += (qv - hi - lo).abs() * kv.abs();
                            }
                            let relu = dot.max(0.0);
                            sc += wh * relu;
                            smag += wh.abs() * relu;
                            kb += wh.abs() * (TC_UNITS * U * p + 2.0 * U * dot.abs());
                            rep += wh.abs() * r;
                            ikb += wh.abs() * gamma(HEAD_DIM) * pq;
                        }
                        let hs = gamma(HEAD_SUM_ROUNDINGS);
                        *o = [
                            sc,
                            (1.0 + hs) * kb + hs * smag,
                            rep,
                            ikb + gamma(HEADS) * smag + U * sc.abs(),
                        ];
                    }
                });
            }
        });
        Rule {
            score: rows.iter().map(|r| r[0]).collect(),
            kernel: rows.iter().map(|r| r[1]).collect(),
            rep: rows.iter().map(|r| r[2]).collect(),
            ik: rows.iter().map(|r| r[3]).collect(),
        }
    }

    /// Rows `0 .. score.len()` by score, highest first, ties to the lower row.
    fn rule_order(score: &[f64]) -> Vec<usize> {
        let mut o: Vec<usize> = (0..score.len()).collect();
        o.sort_by(|&a, &b| score[b].total_cmp(&score[a]).then(a.cmp(&b)));
        o
    }

    /// The order-preserving key of a score — the kernels' `order_key`,
    /// transcribed: `-0` made `+0`, a negative's bits flipped whole, a
    /// non-negative's top bit set.
    fn key_of(v: f32) -> u32 {
        let b = v.to_bits();
        let b = if b == 0x8000_0000 { 0 } else { b };
        if b & 0x8000_0000 != 0 {
            !b
        } else {
            b | 0x8000_0000
        }
    }

    /// The `k` rows of `scores` with the largest keys, ties to the lower row,
    /// in ascending row order: what the top-k pass must write, bit for bit.
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

    /// A selection `ids` against the rule's at every row outside the tie
    /// band: row `t` is left out when `|s_t − s_k| <= β_t + β_k`, `s_k` the
    /// rule's k-th score. Returns the rows that disagree and those left out.
    fn ids_vs_rule(ids: &[u32], score: &[f64], beta: &[f64], k: usize) -> (usize, usize) {
        let order = rule_order(score);
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

    /// A list row: its first `k` entries are distinct rows below `n` in
    /// ascending order, the rest of its `stride` entries still poisoned.
    fn list_shape(row: &[u32], k: usize, n: usize) -> bool {
        row[..k].windows(2).all(|p| p[0] < p[1])
            && row[..k].iter().all(|&i| (i as usize) < n)
            && row[k..].iter().all(|&i| i == LIST_POISON)
    }

    // ------------------------------------------------------------ launches

    /// One indexer launch's buffers, reused across launches of one shape.
    struct Launch {
        q: DeviceBuffer<f32>,
        w: DeviceBuffer<f32>,
        ints: DeviceBuffer<u32>,
        tables: DeviceBuffer<u32>,
        keys: DeviceTensor<u16>,
        scratch: IndexerScratch,
        list: DeviceBuffer<u32>,
        tokens: usize,
        stride: usize,
        n_dims: usize,
    }

    /// What one launch left, read back.
    struct Out {
        q: Vec<f32>,
        w: Vec<f32>,
        scores: Vec<f32>,
        list: Vec<u32>,
        /// The histogram is all zero after the pair.
        hist_zero: bool,
    }

    /// The gate's counts buffer: junk, `n_vis` per token, `top_k`.
    fn ints_words(n_vis: &[u32], top_k: u32) -> Vec<u32> {
        let mut v = vec![0xdead_beef; N_VIS_AT];
        v.extend_from_slice(n_vis);
        v.push(top_k);
        v
    }

    impl Launch {
        #[allow(
            clippy::too_many_arguments,
            reason = "one launch's whole input set, each named at the call"
        )]
        fn new(
            stream: &CudaStream,
            q_raw: &[f32],
            w_raw: &[f32],
            n_vis: &[u32],
            top_k: usize,
            cs: &[f32],
            keys: &[u16],
            stride: usize,
        ) -> Result<Launch, GateError> {
            let tokens = n_vis.len();
            let rows = keys.len() / HEAD_DIM;
            let n_dims = cs.len() / tokens;
            let mut tables = vec![0x7fc0_dead; ROPE_AT];
            tables.extend(cs.iter().map(|v| v.to_bits()));
            Ok(Launch {
                q: DeviceBuffer::from_host(stream, q_raw)?,
                w: DeviceBuffer::from_host(stream, w_raw)?,
                ints: DeviceBuffer::from_host(stream, &ints_words(n_vis, u32::try_from(top_k)?))?,
                tables: DeviceBuffer::from_host(stream, &tables)?,
                keys: DeviceTensor::upload(stream, keys, rows, HEAD_DIM)?,
                scratch: IndexerScratch::new(stream, tokens, rows)?,
                list: DeviceBuffer::from_host(stream, &vec![LIST_POISON; tokens * stride])?,
                tokens,
                stride,
                n_dims,
            })
        }

        fn set_counts(
            &mut self,
            stream: &CudaStream,
            n_vis: &[u32],
            top_k: usize,
        ) -> Result<(), GateError> {
            self.ints
                .copy_from_host(stream, &ints_words(n_vis, u32::try_from(top_k)?))?;
            Ok(())
        }

        fn set_keys(&mut self, stream: &CudaStream, keys: &[u16]) -> Result<(), GateError> {
            self.keys.buf_mut().copy_from_host(stream, keys)?;
            Ok(())
        }

        fn poison(&mut self, stream: &CudaStream) -> Result<(), GateError> {
            let s = &mut self.scratch;
            s.q.copy_from_host(stream, &vec![POISON; s.q.len()])?;
            s.w.copy_from_host(stream, &vec![POISON; s.w.len()])?;
            s.scores
                .copy_from_host(stream, &vec![POISON; s.scores.len()])?;
            self.list
                .copy_from_host(stream, &vec![LIST_POISON; self.list.len()])?;
            Ok(())
        }

        /// The launch's arguments: its own counts, table, keys and outputs.
        fn args(&mut self) -> IndexerArgs<'_> {
            IndexerArgs {
                q: &self.q,
                w: &self.w,
                ints: &self.ints,
                n_vis_at: N_VIS_AT,
                top_k_at: N_VIS_AT + self.tokens,
                tables: &self.tables,
                rope_at: ROPE_AT,
                rope_stride: self.n_dims,
                keys: &self.keys,
                tokens: self.tokens,
                scratch: &mut self.scratch,
                list: &mut self.list,
                stride: self.stride,
            }
        }

        fn enqueue(
            &mut self,
            kernels: &IndexerKernels,
            stream: &CudaStream,
        ) -> Result<(), GpuError> {
            kernels.enqueue(stream, self.args())
        }

        fn read(&self, stream: &CudaStream) -> Result<Out, GateError> {
            stream.synchronize()?;
            Ok(Out {
                q: self.scratch.q.to_host_vec(stream)?,
                w: self.scratch.w.to_host_vec(stream)?,
                scores: self.scratch.scores.to_host_vec(stream)?,
                list: self.list.to_host_vec(stream)?,
                hist_zero: self
                    .scratch
                    .hist
                    .to_host_vec(stream)?
                    .iter()
                    .all(|&c| c == 0),
            })
        }

        fn run(&mut self, kernels: &IndexerKernels, stream: &CudaStream) -> Result<Out, GateError> {
            self.poison(stream)?;
            self.enqueue(kernels, stream)?;
            self.read(stream)
        }
    }

    fn same_out(a: &Out, b: &Out) -> bool {
        bits_equal(&a.q, &b.q)
            && bits_equal(&a.w, &b.w)
            && bits_equal(&a.scores, &b.scores)
            && a.list == b.list
            && a.hist_zero
            && b.hist_zero
    }

    // ---------------------------------------------------------------- run

    /// What every check reads besides its set.
    struct Cx {
        gpu: Gpu,
        kernels: IndexerKernels,
        attn: AttnKernels,
        cand: CandKernels,
        split: Split,
        hp: Hparams,
        yarn: RopeSpec,
        q8: HashMap<String, Q8>,
        q3: HashMap<String, Q3>,
    }

    impl Cx {
        /// Our YaRN table at `pos` (the image's `YarnForward`) and ik's
        /// (ggml's recipe for a 128-value head, of which the tail reads the
        /// first `n_dims`).
        fn tables(&self, pos: u32) -> Result<(Vec<f32>, Vec<f32>), GateError> {
            let mut ours = Vec::with_capacity(self.hp.rope_dims);
            RopeTable::new(&self.yarn)?.push(pos, Direction::Forward, &mut ours);
            let ik = ggml_rope_cache(&self.yarn, pos, HEAD_DIM, Direction::Forward)
                .into_iter()
                .take(self.hp.rope_dims)
                .collect();
            Ok((ours, ik))
        }

        fn ensure_q8(&mut self, name: &str) -> Result<(), GateError> {
            if !self.q8.contains_key(name) {
                let w = load_q8(&self.split, self.gpu.stream(), name)?;
                self.q8.insert(name.to_string(), w);
            }
            Ok(())
        }

        fn ensure_q3(&mut self, name: &str) -> Result<(), GateError> {
            if !self.q3.contains_key(name) {
                let w = load_q3(&self.split, self.gpu.stream(), name)?;
                self.q3.insert(name.to_string(), w);
            }
            Ok(())
        }

        /// `name` in the file's type: q8_0 or q3_K, the two whose rules the
        /// gate transcribes; any other is refused with the tensor named.
        fn ensure_q8_or_q3(&mut self, name: &str) -> Result<(), GateError> {
            let (_, t) = self
                .split
                .find(name)
                .ok_or_else(|| format!("{name} is not in the model file"))?;
            match t.ty {
                GgmlType::Q8_0 => self.ensure_q8(name),
                GgmlType::Q3_K => self.ensure_q3(name),
                ty => Err(format!(
                    "{name} is {ty:?} {:?}, want Q8_0 or Q3_K: the gate has no rule for it",
                    t.dims
                )
                .into()),
            }
        }
    }

    /// Our q3_K gemv of one column `x` at m = 1: its q8_1 form, then
    /// `q3k_gemv` — the engine's dense path for a q3_K projection.
    fn q3k_gemv(cx: &Cx, w: &Q3, x: &[f32]) -> Result<Vec<f32>, GateError> {
        let stream = cx.gpu.stream();
        let x_dev = DeviceBuffer::from_host(stream, x)?;
        let mut act = Q8Act::with_k(stream, 1, w.host.k)?;
        let mut y = DeviceBuffer::<f32>::zeroed(stream, w.host.rows)?;
        cx.gpu.enqueue_quantize_q8_1(&x_dev, &mut act)?;
        cx.gpu.enqueue_gemv_q3k(&w.dev, &act, &mut y)?;
        stream.synchronize()?;
        Ok(y.to_host_vec(stream)?)
    }

    pub fn run() -> Result<(), GateError> {
        let model = ref_model_path()?;
        let split = Split::open(&model).map_err(|e| format!("open {}: {e}", model.display()))?;
        if split.architecture() != Some(Arch::Deepseek41.name()) {
            return Err(format!(
                "{} is a {:?} model, want {}",
                model.display(),
                split.architecture(),
                Arch::Deepseek41.name()
            )
            .into());
        }
        let hp = Hparams::read(&split)?;
        let (_, yarn) = rope_specs(&hp)?;
        let gpu = Gpu::new()?;
        let kernels = IndexerKernels::load(gpu.context(), &hp)?;
        let attn = AttnKernels::load(gpu.context())?;
        let cand = CandKernels::load(gpu.context())?;
        let emitting: Vec<usize> = (0..hp.layers.len())
            .filter(|&l| hp.layers[l].indexer)
            .collect();
        println!(
            "gate_deepseek41_index: model {} device {} — {} heads of {}, rope {} dims, weights scale \
             {:e}, file top_k {}, emitting layers {}",
            model.display(),
            gpu.device_name()?,
            hp.indexer.n_head,
            hp.indexer.head_dim,
            hp.rope_dims,
            kernels.scale(),
            hp.indexer.top_k,
            list(&emitting)
        );
        let mut cx = Cx {
            gpu,
            kernels,
            attn,
            cand,
            split,
            hp,
            yarn,
            q8: HashMap::new(),
            q3: HashMap::new(),
        };
        let mut ok = no_local_depot(&["ds41_indexer_score", "ds41_indexer_topk"])?;
        let oracle = for_arch(Arch::Deepseek41)?;
        let mut layers = 0usize;
        let mut failed = 0usize;
        for name in [D1, D1_UNFUSED, D2, D2_UNFUSED] {
            let man = oracle.open_named(name)?;
            let (n, f) = gate_set(&mut cx, name, &man)?;
            layers += n;
            failed += f;
        }
        let man = oracle.open_named(D1N)?;
        ok &= identity_set(&mut cx, D1N, &man)?;
        ok &= identity_attention(&cx)?;
        for (n, seed) in [(16_384usize, 11u32), (32_768, 12)] {
            ok &= planted_case(&cx, n, seed)?;
        }
        ok &= clustered_case(&cx, 32_768, 13)?;
        ok &= replay(&cx)?;
        ok &= cand_set(&mut cx)?;
        ok &= failed == 0;
        println!(
            "gate_deepseek41_index: {layers} layer cases, {failed} failed; identity, attention, depth \
             replay and candidate-mask checks above — {}",
            verdict(ok)
        );
        if ok { Ok(()) } else { Err(checks_failed()) }
    }

    // ------------------------------------------------------------ the sets

    /// One decode-step set's position, selection width and plan.
    struct SetInfo {
        pos: u32,
        top_k: usize,
        fused: bool,
        ctx: u64,
        planner: Planner,
        plan: StepPlan,
    }

    fn set_info(cx: &Cx, man: &RefManifest) -> Result<SetInfo, GateError> {
        let (pos0, tokens, before) = man.step()?;
        if tokens.len() != 1 {
            return Err(format!(
                "{}: a step of {} tokens, want one",
                man.dir.display(),
                tokens.len()
            )
            .into());
        }
        let flags = man
            .header
            .flags
            .as_deref()
            .ok_or_else(|| format!("{}: no # flags", man.dir.display()))?;
        let top_k = top_k_override(flags)?.unwrap_or(cx.hp.indexer.top_k);
        let hp = cx.hp.clone().with_indexer_top_k(top_k);
        let ctx = man
            .header
            .ctx
            .ok_or_else(|| format!("{}: no -c in # flags", man.dir.display()))?;
        let planner = Planner::from_file(&cx.split, &hp, ctx)?;
        let mut plan = StepPlan::default();
        planner.plan_into(tokens, pos0, before, &mut plan)?;
        Ok(SetInfo {
            pos: pos0,
            top_k: hp.indexer.top_k,
            fused: man.header.fused_idx_topk.unwrap_or(true),
            ctx,
            planner,
            plan,
        })
    }

    /// A layer's visible rows and its stream's cache height.
    fn layer_rows(info: &SetInfo, l: usize) -> Result<(usize, usize), GateError> {
        let s = info
            .planner
            .layer_stream(l)
            .ok_or_else(|| format!("layer {l} runs the indexer and has no stream"))?;
        let st = &info.plan.streams[s];
        let n_vis = st.n_visible[0] as usize;
        let rows = usize::try_from(info.ctx)?.div_ceil(st.ratio as usize);
        Ok((n_vis, rows))
    }

    /// Gate every emitting layer of one selecting set: (layer cases, failed).
    fn gate_set(cx: &mut Cx, label: &str, man: &RefManifest) -> Result<(usize, usize), GateError> {
        let info = set_info(cx, man)?;
        println!(
            "set {label}: {} (build {}) — position {}, top_k {}, {} top-k, context {}",
            man.dir.display(),
            man.build.as_deref().unwrap_or("-"),
            info.pos,
            info.top_k,
            if info.fused { "fused" } else { "unfused" },
            info.ctx
        );
        let (mut n, mut failed) = (0usize, 0usize);
        for l in 0..cx.hp.layers.len() {
            if cx.hp.layers[l].indexer {
                n += 1;
                failed += usize::from(!gate_layer(cx, label, man, &info, l)?);
            }
        }
        Ok((n, failed))
    }

    /// The tensor row `name`/0, of op `op`.
    fn node<'a>(
        man: &'a RefManifest,
        name: &str,
        op: &str,
    ) -> Result<(usize, &'a RefRow), GateError> {
        man.tensor_at(name, 0)
            .ok()
            .filter(|(_, r)| r.op == op)
            .ok_or_else(|| format!("no {op} node {name}/0").into())
    }

    /// A row's plain file as f32, non-finite values kept (the masked scores).
    fn raw_f32(man: &RefManifest, row: &RefRow) -> Result<Vec<f32>, GateError> {
        let path = man.dir.join(row.file_name());
        let raw = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        if raw.len() as u64 != 4 * row.count() {
            return Err(format!(
                "{}: {} bytes for {} values",
                path.display(),
                raw.len(),
                row.count()
            )
            .into());
        }
        Ok(raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect())
    }

    fn gate_layer(
        cx: &mut Cx,
        label: &str,
        man: &RefManifest,
        info: &SetInfo,
        l: usize,
    ) -> Result<bool, GateError> {
        let (n_vis, rows) = layer_rows(info, l)?;
        let k = info.top_k;
        if n_vis <= k {
            return Err(format!(
                "{label} layer {l}: {n_vis} visible rows select nothing at top_k {k}"
            )
            .into());
        }
        // The dump's chain.
        let (at_q, q_row) = node(man, &format!("lid_q-{l}"), "MUL_MAT")?;
        let q_name = format!("blk.{l}.indexer.attn_q_b.weight");
        if q_row.src0.as_deref() != Some(q_name.as_str()) {
            return Err(format!("lid_q-{l} multiplies {:?}, want {q_name}", q_row.src0).into());
        }
        let (_, xq_row) = man.last_before(at_q, q_row.src1.as_deref())?;
        let (at_w, w_row) = node(man, &format!("lid_weights-{l}"), "MUL_MAT")?;
        let w_name = format!("blk.{l}.indexer.proj.weight");
        if w_row.src0.as_deref() != Some(w_name.as_str()) {
            return Err(
                format!("lid_weights-{l} multiplies {:?}, want {w_name}", w_row.src0).into(),
            );
        }
        let (_, xw_row) = man.last_before(at_w, w_row.src1.as_deref())?;
        let scale_row = man.tensors[at_w..]
            .iter()
            .find(|r| r.op == "SCALE" && r.src0.as_deref() == Some(w_row.name.as_str()))
            .ok_or_else(|| format!("no SCALE of lid_weights-{l}"))?;
        let (_, roped_row) = node(man, &format!("indexer_q-{l}"), "ROPE")?;
        let (_, hada_row) = node(man, &format!("lid_q_hadamard-{l}"), "HADAMARD")?;
        let (_, k_row) = node(man, &format!("lid_k-{l}"), "VIEW")?;
        let dir = &man.dir;
        let x_q = ref_tensor_logical_in(dir, xq_row)?;
        let x_w = ref_tensor_logical_in(dir, xw_row)?;
        let d_q = ref_tensor_logical_in(dir, q_row)?;
        let d_roped = ref_tensor_logical_in(dir, roped_row)?;
        let d_hada = ref_tensor_logical_in(dir, hada_row)?;
        let d_w = ref_tensor_logical_in(dir, w_row)?;
        let d_scaled = ref_tensor_logical_in(dir, scale_row)?;
        let nq = HEADS * HEAD_DIM;
        if [d_q.len(), d_roped.len(), d_hada.len()] != [nq; 3]
            || [d_w.len(), d_scaled.len()] != [HEADS; 2]
        {
            return Err(
                format!("layer {l}: the query chain is not {HEADS} heads of {HEAD_DIM}").into(),
            );
        }
        let kv = widened_f16_rows_in(dir, k_row)?;
        if kv.len() < n_vis * HEAD_DIM {
            return Err(format!(
                "lid_k-{l} holds {} rows, the plan sees {n_vis}",
                kv.len() / HEAD_DIM
            )
            .into());
        }
        let ik_ids: Vec<u32> = ref_ints(
            man,
            &format!("lid_top_k-{l}"),
            0,
            RowKind::Tensor,
            Layout::Flat,
        )?
        .into_iter()
        .map(u32::try_from)
        .collect::<Result<_, _>>()?;
        let ik_shape = ik_ids.len() == k && {
            let mut s = ik_ids.clone();
            s.sort_unstable();
            s.dedup();
            s.len() == k && s.iter().all(|&i| (i as usize) < n_vis)
        };
        // The unfused set's scores, and its mask: the rows from n_vis on.
        let unfused = if info.fused {
            None
        } else {
            let (_, s_row) = node(man, &format!("lid_score-{l}"), "CONT")?;
            let (_, m_row) = node(man, &format!("lid_score_masked-{l}"), "ADD")?;
            let score = ref_tensor_logical_in(dir, s_row)?;
            let masked = raw_f32(man, m_row)?;
            let mask_ok = masked.len() == score.len()
                && n_vis <= score.len()
                && bits_equal(&masked[..n_vis], &score[..n_vis])
                && masked[n_vis..].iter().all(|&v| v == f32::NEG_INFINITY);
            Some((score, mask_ok))
        };

        // The key source: the dump's rows below n_vis, NaN above.
        let mut keys = vec![NAN_F16; rows * HEAD_DIM];
        keys[..n_vis * HEAD_DIM].copy_from_slice(&kv[..n_vis * HEAD_DIM]);
        let (cs, cs_ik) = cx.tables(info.pos)?;
        let tables_same = bits_equal(&cs, &cs_ik);
        let scale = cx.kernels.scale();

        // Run A's gemvs, and ik's rules for them: the query's in the file's
        // type of attn_q_b, each row's band and ik's value. A q3_K query
        // gemv is also pinned by itself, per row: against the exact dot of
        // our q8_1 values within KERNEL_BAND (gate_p1's form), and against
        // the dump within its row's band — through the rope and the
        // transform a row's defect is diluted by the other rows' bands.
        cx.ensure_q8_or_q3(&q_name)?;
        cx.ensure_q3(&w_name)?;
        let cx = &*cx;
        let (q_raw_a, dl, q_ik, q_dot, q_gemv) = if let Some(q8) = cx.q8.get(&q_name) {
            let q8r = q8_rows(q8, &x_q);
            let stream = cx.gpu.stream();
            let x_dev = DeviceBuffer::from_host(stream, &x_q)?;
            let mut y = DeviceBuffer::<f32>::zeroed(stream, q8.rows)?;
            cx.gpu
                .q8f32()
                .enqueue_q8_0_gemv(stream, &q8.qs, &q8.d, &x_dev, 1, &mut y)?;
            stream.synchronize()?;
            let dl: Vec<f64> = q8r.iter().map(|r| r.band).collect();
            let ik: Vec<f32> = q8r.iter().map(|r| r.ik).collect();
            (y.to_host_vec(stream)?, dl, ik, "q8_2_dot", None)
        } else {
            let q3 = &cx.q3[&q_name];
            let rows = act_rule::q3k_rows(&q3.host, &x_q, q3.host.rows);
            let ik = act_rule::ik_q3k_rows(&q3.host, &x_q, q3.host.rows);
            let y = q3k_gemv(cx, q3, &x_q)?;
            let ours: Vec<f32> = rows.iter().map(|r| r.ours as f32).collect();
            let rel = max_rel_err(&y, &ours).unwrap_or(f32::INFINITY);
            let over = y
                .iter()
                .zip(&d_q)
                .zip(&rows)
                .filter(|&((&g, &v), r)| outside((f64::from(g) - f64::from(v)).abs(), r.band))
                .count();
            let dl = rows.iter().map(|r| r.band).collect();
            (y, dl, ik, "q3k_dot", Some((rel, over)))
        };
        let q_gemv_ok = q_gemv.is_none_or(|(rel, over)| rel <= KERNEL_BAND && over == 0);
        let q_gemv_line = q_gemv.map_or_else(String::new, |(rel, over)| {
            format!(" q_gemv_vs_ours_rel={rel:.3e} q_gemv_over={over}/{nq}")
        });
        let q3 = &cx.q3[&w_name];
        let q3r = act_rule::q3k_rows(&q3.host, &x_w, q3.host.rows);
        let w_raw_a = q3k_gemv(cx, q3, &x_w)?;
        if q_raw_a.len() != nq || w_raw_a.len() != HEADS {
            return Err(format!(
                "layer {l}: the projections are {} and {} wide",
                q_raw_a.len(),
                w_raw_a.len()
            )
            .into());
        }

        // The two runs, and a rerun of A.
        let stream = cx.gpu.stream();
        let stride = k + LIST_SLACK;
        let nv = [u32::try_from(n_vis)?];
        let mut la = Launch::new(stream, &q_raw_a, &w_raw_a, &nv, k, &cs, &keys, stride)?;
        let a = la.run(&cx.kernels, stream)?;
        let rerun = same_out(&a, &la.run(&cx.kernels, stream)?);
        let mut lb = Launch::new(stream, &d_q, &d_w, &nv, k, &cs, &keys, stride)?;
        let b = lb.run(&cx.kernels, stream)?;

        // (i) Layer 1: run A against our rules on its own gemv output.
        let host_q = query(&q_raw_a, &cs);
        let q_rule = bits_equal(&a.q, &host_q);
        let host_w: Vec<f32> = w_raw_a.iter().map(|&v| v * scale).collect();
        let w_rule = bits_equal(&a.w, &host_w);
        // Layer 2: ik's rules against the dump.
        let ik_q_same = q_ik
            .iter()
            .zip(&d_q)
            .filter(|(r, v)| r.to_bits() == v.to_bits())
            .count();
        let ik_rope_same = d_q
            .chunks(HEAD_DIM)
            .zip(d_roped.chunks(HEAD_DIM))
            .map(|(x, y)| usize::from(bits_equal(&rotate(x, &cs_ik), y)))
            .sum::<usize>();
        let mut ht_scale_is = true;
        let ik_hada_same = d_roped
            .chunks(HEAD_DIM)
            .zip(d_hada.chunks(HEAD_DIM))
            .map(|(x, y)| {
                let mut v = x.to_vec();
                ht_scale_is &= fast_ht(&mut v).to_bits() == HT_SCALE.to_bits();
                usize::from(bits_equal(&v, y))
            })
            .sum::<usize>();
        let ik_w_over = q3r
            .iter()
            .zip(&d_w)
            .filter(|(r, v)| (f64::from(**v) - r.ik).abs() > r.ik_band)
            .count();
        let ik_scale = d_w
            .iter()
            .zip(&d_scaled)
            .all(|(&v, &s)| (v * scale).to_bits() == s.to_bits());
        // Layer 3: run A against the dump.
        let qb = query_band(&q_raw_a, &dl, &cs);
        let (mut q_over, mut q_worst) = (0usize, 0.0f64);
        for (i, (&x, &y)) in a.q.iter().zip(&d_hada).enumerate() {
            let d = (f64::from(x) - f64::from(y)).abs();
            let bd = qb[i / HEAD_DIM];
            q_over += usize::from(outside(d, bd));
            q_worst = q_worst.max(d / bd);
        }
        let (mut w_over, mut w_worst) = (0usize, 0.0f64);
        for ((&x, &y), r) in a.w.iter().zip(&d_scaled).zip(&q3r) {
            let d = (f64::from(x) - f64::from(y)).abs();
            let bd = f64::from(scale) * r.band;
            w_over += usize::from(outside(d, bd));
            w_worst = w_worst.max(d / bd);
        }
        let q_rel = max_rel_err(&a.q, &d_hada).unwrap_or(f32::INFINITY);
        // Run B: our rope, transform and scale on ik's projections are ik's.
        let q_b = bits_equal(&b.q, &d_hada);
        let w_b = bits_equal(&b.w, &d_scaled);
        let i_ok = q_rule
            && q_gemv_ok
            && w_rule
            && ik_q_same == nq
            && ik_rope_same == HEADS
            && ik_hada_same == HEADS
            && ht_scale_is
            && ik_w_over == 0
            && ik_scale
            && tables_same
            && q_over == 0
            && w_over == 0
            && q_b
            && w_b;

        // (ii) The scores against our rule, and run B's against the dump.
        let mut ii_ok = true;
        let mut ii_line = String::new();
        let mut rules = Vec::with_capacity(2);
        for (tag, out) in [("A", &a), ("B", &b)] {
            let r = rule(&out.q[..nq], &out.w[..HEADS], &keys, n_vis);
            let got = &out.scores[..n_vis];
            let (mut over, mut worst) = (0usize, 0.0f64);
            for ((&x, &s), &bd) in got.iter().zip(&r.score).zip(&r.kernel) {
                let d = (f64::from(x) - s).abs();
                over += usize::from(outside(d, bd));
                worst = worst.max(d / bd);
            }
            let rule_f32: Vec<f32> = r.score.iter().map(|&s| s as f32).collect();
            let rel = max_rel_err(got, &rule_f32).unwrap_or(f32::INFINITY);
            let past = out.scores[n_vis..]
                .iter()
                .all(|v| v.to_bits() == POISON.to_bits());
            ii_ok &= over == 0 && past;
            ii_line.push_str(&format!(
                " {tag}:rule_over={over}/{n_vis} rule_worst={worst:.3} rule_rel={rel:.2e} past_n_vis_untouched={past}"
            ));
            rules.push(r);
        }
        let rule_b = &rules[1];
        if let Some((score, mask_ok)) = &unfused {
            let (mut over, mut worst) = (0usize, 0.0f64);
            for t in 0..n_vis {
                let d = (f64::from(b.scores[t]) - f64::from(score[t])).abs();
                let bd = rule_b.kernel[t] + rule_b.rep[t] + rule_b.ik[t];
                over += usize::from(outside(d, bd));
                worst = worst.max(d / bd);
            }
            let rel = max_rel_err(&b.scores[..n_vis], &score[..n_vis]).unwrap_or(f32::INFINITY);
            ii_ok &= over == 0 && *mask_ok;
            ii_line.push_str(&format!(
                " B:dump_over={over}/{n_vis} dump_worst={worst:.3} dump_rel={rel:.2e} mask_hides_from_n_vis={mask_ok}"
            ));
        }

        // (iii) The ids.
        let beta: Vec<f64> = (0..n_vis)
            .map(|t| rule_b.kernel[t] + rule_b.rep[t] + rule_b.ik[t])
            .collect();
        let (dev_off, band_rows) = ids_vs_rule(&b.list[..k], &rule_b.score, &beta, k);
        let (ik_off, _) = ids_vs_rule(&ik_ids, &rule_b.score, &beta, k);
        let mut dev_set: Vec<u32> = b.list[..k].to_vec();
        dev_set.sort_unstable();
        let mut ik_set = ik_ids.clone();
        ik_set.sort_unstable();
        let symdiff = dev_set
            .iter()
            .filter(|i| ik_set.binary_search(i).is_err())
            .count()
            + ik_set
                .iter()
                .filter(|i| dev_set.binary_search(i).is_err())
                .count();
        let exact_a = a.list[..k] == exact_top(&a.scores[..n_vis], k)[..];
        let exact_b = b.list[..k] == exact_top(&b.scores[..n_vis], k)[..];
        let shape = list_shape(&a.list, k, n_vis) && list_shape(&b.list, k, n_vis);
        let iii_ok = dev_off == 0 && ik_off == 0 && ik_shape && exact_a && exact_b && shape;
        let hist = a.hist_zero && b.hist_zero;

        let pass = i_ok && ii_ok && iii_ok && hist && rerun;
        println!(
            "layer set={label} layer={l} n_vis={n_vis} rows={rows} top_k={k} — (i) A:q_rule_bits={q_rule} \
             w_rule_bits={w_rule}{q_gemv_line} q_over={q_over}/{nq} q_worst={q_worst:.3} q_rel={q_rel:.2e} \
             w_over={w_over}/{HEADS} w_worst={w_worst:.3} B:q_is_ik={q_b} w_is_ik={w_b} ik_sim: \
             {q_dot}={ik_q_same}/{nq} rope={ik_rope_same}/{HEADS} hadamard={ik_hada_same}/{HEADS} \
             ht_scale={ht_scale_is} q3k_over={ik_w_over}/{HEADS} scale_bits={ik_scale} \
             table_is_ggml={tables_same} — (ii){ii_line} — (iii) B_vs_rule_off={dev_off} \
             ik_vs_rule_off={ik_off} tie_band_rows={band_rows} B_vs_ik_symdiff={symdiff} \
             ik_ids_shape={ik_shape} exact_top_A={exact_a} exact_top_B={exact_b} list_shape={shape} \
             hist_zero={hist} bit_identical_rerun={rerun} {}",
            verdict(pass)
        );
        Ok(pass)
    }

    // ------------------------------------------------------------ identity

    /// (iv) Every emitting layer of the identity set: the list is
    /// `0..n_vis`, and nothing else is written — the scores, query and
    /// weights stay poisoned and the histogram zero.
    fn identity_set(cx: &mut Cx, label: &str, man: &RefManifest) -> Result<bool, GateError> {
        let info = set_info(cx, man)?;
        println!(
            "set {label}: {} (build {}) — position {}, top_k {}, context {}",
            man.dir.display(),
            man.build.as_deref().unwrap_or("-"),
            info.pos,
            info.top_k,
            info.ctx
        );
        let k = info.top_k;
        let stride = k + LIST_SLACK;
        let (cs, _) = cx.tables(info.pos)?;
        let mut ok = true;
        for l in 0..cx.hp.layers.len() {
            if !cx.hp.layers[l].indexer {
                continue;
            }
            let (n_vis, rows) = layer_rows(&info, l)?;
            if n_vis > k {
                return Err(
                    format!("{label} layer {l}: {n_vis} visible rows at top_k {k} select").into(),
                );
            }
            let keys: Vec<u16> = activations(HEAD_DIM, rows, 3 + l as u32)
                .into_iter()
                .map(f32_to_f16_bits)
                .collect();
            let q_raw = activations(HEADS * HEAD_DIM, 1, 5);
            let w_raw = activations(HEADS, 1, 6);
            let stream = cx.gpu.stream();
            let nv = [u32::try_from(n_vis)?];
            let mut launch = Launch::new(stream, &q_raw, &w_raw, &nv, k, &cs, &keys, stride)?;
            let out = launch.run(&cx.kernels, stream)?;
            let identity = out.list[..n_vis]
                .iter()
                .enumerate()
                .all(|(i, &v)| v as usize == i);
            let rest = out.list[n_vis..].iter().all(|&v| v == LIST_POISON);
            let untouched = [&out.q, &out.w, &out.scores]
                .iter()
                .all(|v| v.iter().all(|x| x.to_bits() == POISON.to_bits()));
            let pass = identity && rest && untouched && out.hist_zero;
            ok &= pass;
            println!(
                "identity set={label} layer={l} n_vis={n_vis} top_k={k} rows={rows} list_is_0..n_vis={identity} \
                 rest_untouched={rest} score_pass_wrote_nothing={untouched} hist_zero={} {}",
                out.hist_zero,
                verdict(pass)
            );
        }
        Ok(ok)
    }

    /// (iv) The selected-row attention over an identity list against the
    /// prefix attention, bit for bit: synthetic query, ring and stream
    /// (NaN past the count), the list from the top-k pass at `n <= top_k`.
    fn identity_attention(cx: &Cx) -> Result<bool, GateError> {
        let stream = cx.gpu.stream();
        let k = cx.hp.indexer.top_k;
        let stride = k + LIST_SLACK;
        // A multiple of the attention's 16-row tile.
        let heads = 16;
        let (win_rows, win_live) = (128usize, 64u32);
        let comp_rows = k + 9;
        let mut ok = true;
        for n in [302usize, k] {
            let keys: Vec<u16> = activations(HEAD_DIM, comp_rows, 21)
                .into_iter()
                .map(f32_to_f16_bits)
                .collect();
            let (cs, _) = cx.tables(4 * u32::try_from(n)?)?;
            let nv = [u32::try_from(n)?];
            let mut launch = Launch::new(
                stream,
                &activations(HEADS * HEAD_DIM, 1, 22),
                &activations(HEADS, 1, 23),
                &nv,
                k,
                &cs,
                &keys,
                stride,
            )?;
            let sel = launch.run(&cx.kernels, stream)?;
            let identity = sel.list[..n]
                .iter()
                .enumerate()
                .all(|(i, &v)| v as usize == i);

            let q = activations(LATENT, heads, 24);
            let ring: Vec<u16> = activations(LATENT, win_rows, 25)
                .into_iter()
                .map(f32_to_f16_bits)
                .collect();
            let mut comp: Vec<u16> = activations(LATENT, comp_rows, 26)
                .into_iter()
                .map(f32_to_f16_bits)
                .collect();
            comp[n * LATENT..].fill(NAN_F16);
            let sinks = activations(heads, 1, 27);
            let q_dev = DeviceBuffer::from_host(stream, &q)?;
            let ring_dev = DeviceTensor::upload(stream, &ring, win_rows, LATENT)?;
            let comp_dev = DeviceTensor::upload(stream, &comp, comp_rows, LATENT)?;
            let vis = DeviceBuffer::from_host(stream, &[win_live, u32::try_from(n)?])?;
            let sinks_dev = DeviceBuffer::from_host(stream, &sinks)?;
            let segs = attn::segments(win_rows, comp_rows.max(stride));
            let mut ys = Vec::with_capacity(2);
            for selected in [false, true] {
                let mut part_v = DeviceBuffer::from_host(
                    stream,
                    &vec![POISON; attn::partials_v_len(heads, segs)],
                )?;
                let mut part_ms = DeviceBuffer::from_host(
                    stream,
                    &vec![POISON; attn::partials_ms_len(heads, segs)],
                )?;
                let mut y = DeviceBuffer::from_host(stream, &vec![POISON; heads * LATENT])?;
                cx.attn.enqueue(
                    stream,
                    AttnArgs {
                        q: &q_dev,
                        window: &ring_dev,
                        compressed: Some(&comp_dev),
                        selected: selected.then_some(SelectedRows {
                            rows: &launch.list,
                            stride,
                        }),
                        vis: &vis,
                        sinks: &sinks_dev,
                        scale: 0.041_666_668,
                        tokens: 1,
                        heads,
                        part_v: &mut part_v,
                        part_ms: &mut part_ms,
                        y: &mut y,
                        fault: cx.gpu.unlabelled_sink(),
                    },
                )?;
                stream.synchronize()?;
                ys.push(y.to_host_vec(stream)?);
            }
            let finite = ys[0].iter().all(|v| v.is_finite());
            let same = bits_equal(&ys[0], &ys[1]);
            let pass = identity && finite && same;
            ok &= pass;
            println!(
                "identity-attention n={n} top_k={k} stride={stride} window={win_live}/{win_rows} rows={comp_rows} \
                 list_is_0..n={identity} prefix_finite={finite} sel_bit_identical_to_prefix={same} {}",
                verdict(pass)
            );
        }
        Ok(ok)
    }

    // ------------------------------------------------------------- depth

    /// Synthetic inputs of one depth case: `n` visible rows of an `n + 37`
    /// row cache (NaN past `n`), a query, weights, the table at a far
    /// position.
    struct Synth {
        q_raw: Vec<f32>,
        w_raw: Vec<f32>,
        cs: Vec<f32>,
        keys: Vec<u16>,
    }

    fn synth(cx: &Cx, n: usize, seed: u32) -> Result<Synth, GateError> {
        let rows = n + 37;
        let mut keys: Vec<u16> = activations(HEAD_DIM, rows, seed)
            .into_iter()
            .map(f32_to_f16_bits)
            .collect();
        keys[n * HEAD_DIM..].fill(NAN_F16);
        let q_raw = activations(HEADS * HEAD_DIM, 1, seed + 100)
            .into_iter()
            .map(|v| 4.0 * v)
            .collect();
        let w_raw = activations(HEADS, 1, seed + 200)
            .into_iter()
            .map(|v| 16.0 * v)
            .collect();
        let (cs, _) = cx.tables(4 * u32::try_from(n)? - 1)?;
        Ok(Synth {
            q_raw,
            w_raw,
            cs,
            keys,
        })
    }

    /// (v) A depth case with a planted tie: after one launch gives the
    /// kernel's query and weights, the key of the row at rule rank `ρ` is
    /// copied onto `GROUP − 1` rows ranked far below the threshold (two
    /// before it and two after it where the rows allow), so the `GROUP` rows
    /// score bit-identically. `ρ` runs up from `k − GROUP + 1` until the kernel's
    /// own scores put the group across the threshold — `need` of its rows
    /// fit, `0 < need < GROUP` — and then exactly the lowest `need` rows
    /// must be taken. At depth the rule's neighbours of the threshold crowd
    /// within the band, so the selection is held to the rule outside the tie
    /// band ([`ids_vs_rule`]) and to the kernel's own scores exactly.
    fn planted_case(cx: &Cx, n: usize, seed: u32) -> Result<bool, GateError> {
        let stream = cx.gpu.stream();
        let k = cx.hp.indexer.top_k;
        let stride = k + LIST_SLACK;
        let s = synth(cx, n, seed)?;
        let nv = [u32::try_from(n)?];
        let mut launch = Launch::new(stream, &s.q_raw, &s.w_raw, &nv, k, &s.cs, &s.keys, stride)?;
        let first = launch.run(&cx.kernels, stream)?;
        let (q, w) = (&first.q[..HEADS * HEAD_DIM], &first.w[..HEADS]);
        let order = rule_order(&rule(q, w, &s.keys, n).score);
        let mut far: Vec<usize> = order[k + 64..].to_vec();
        far.sort_unstable();
        let mut plant = None;
        for &star in &order[k + 1 - GROUP..k] {
            let before: Vec<usize> = far.iter().copied().filter(|&r| r < star).collect();
            let after: Vec<usize> = far.iter().copied().filter(|&r| r > star).collect();
            let mut copies: Vec<usize> = before.iter().take(2).copied().collect();
            copies.extend(after.iter().rev().take(GROUP - 1 - copies.len()));
            if copies.len() < GROUP - 1 {
                copies.extend(before.iter().skip(2).take(GROUP - 1 - copies.len()));
            }
            let mut keys = s.keys.clone();
            for &c in &copies {
                keys.copy_within(star * HEAD_DIM..(star + 1) * HEAD_DIM, c * HEAD_DIM);
            }
            launch.set_keys(stream, &keys)?;
            let out = launch.run(&cx.kernels, stream)?;
            let ks = key_of(out.scores[star]);
            let above = out.scores[..n].iter().filter(|&&v| key_of(v) > ks).count();
            if above < k && k - above < GROUP {
                plant = Some((star, copies, keys, out, k - above));
                break;
            }
        }
        let Some((star, copies, keys, out, need)) = plant else {
            println!(
                "depth-planted n={n} top_k={k} no planted group straddles the kernel's threshold FAIL"
            );
            return Ok(false);
        };
        let r = rule(q, w, &keys, n);
        let again = launch.run(&cx.kernels, stream)?;
        let rerun =
            same_out(&out, &again) && bits_equal(&out.q, &first.q) && bits_equal(&out.w, &first.w);

        let got = &out.scores[..n];
        let (mut over, mut worst) = (0usize, 0.0f64);
        for ((&x, &sv), &bd) in got.iter().zip(&r.score).zip(&r.kernel) {
            let d = (f64::from(x) - sv).abs();
            over += usize::from(outside(d, bd));
            worst = worst.max(d / bd);
        }
        let past = out.scores[n..]
            .iter()
            .all(|v| v.to_bits() == POISON.to_bits());
        let group_same = copies
            .iter()
            .all(|&c| got[c].to_bits() == got[star].to_bits());
        let exact = out.list[..k] == exact_top(got, k)[..];
        let (rule_off, band_rows) = ids_vs_rule(&out.list[..k], &r.score, &r.kernel, k);
        let mut group: Vec<usize> = copies.clone();
        group.push(star);
        group.sort_unstable();
        let taken: Vec<usize> = group
            .iter()
            .copied()
            .filter(|&g| out.list[..k].contains(&(g as u32)))
            .collect();
        let lowest = taken == group[..need];
        let shape = list_shape(&out.list, k, n);
        let pass = over == 0
            && past
            && group_same
            && exact
            && rule_off == 0
            && lowest
            && shape
            && out.hist_zero
            && rerun;
        println!(
            "depth-planted n={n} rows={} top_k={k} tie_group={} need={need} taken={} rule_over={over}/{n} \
             rule_worst={worst:.3} past_n_vis_untouched={past} group_bit_identical={group_same} \
             exact_top={exact} rule_off={rule_off} tie_band_rows={band_rows} ties_to_lower_rows={lowest} list_shape={shape} \
             hist_zero={} bit_identical_rerun={rerun} {}",
            keys.len() / HEAD_DIM,
            list(&group),
            list(&taken),
            out.hist_zero,
            verdict(pass)
        );
        Ok(pass)
    }

    /// (v) A clustered depth case: every row is one key with four dims
    /// moved by at most eight f16 steps, so the scores share their top bits,
    /// the refining passes see thousands of candidates, and many rows tie
    /// exactly. The list must be the exact top-k of the kernel's scores, and
    /// the scores within the band of our rule.
    fn clustered_case(cx: &Cx, n: usize, seed: u32) -> Result<bool, GateError> {
        let stream = cx.gpu.stream();
        let k = cx.hp.indexer.top_k;
        let stride = k + LIST_SLACK;
        let s = synth(cx, n, seed)?;
        let base: Vec<u16> = activations(HEAD_DIM, 1, seed + 300)
            .into_iter()
            .map(|v| f32_to_f16_bits(0.5 + 0.25 * v.abs()))
            .collect();
        let jitter = activations(4, n, seed + 400);
        let mut keys = s.keys.clone();
        for t in 0..n {
            let row = &mut keys[t * HEAD_DIM..(t + 1) * HEAD_DIM];
            row.copy_from_slice(&base);
            for (j, &u) in jitter[4 * t..4 * t + 4].iter().enumerate() {
                let step = (u * 8.0).round() as i32;
                let d = 31 * j + 7;
                row[d] = u16::try_from(i32::from(row[d]) + step)?;
            }
        }
        let nv = [u32::try_from(n)?];
        let mut launch = Launch::new(stream, &s.q_raw, &s.w_raw, &nv, k, &s.cs, &keys, stride)?;
        let out = launch.run(&cx.kernels, stream)?;
        let got = &out.scores[..n];
        let r = rule(&out.q[..HEADS * HEAD_DIM], &out.w[..HEADS], &keys, n);
        let (mut over, mut worst) = (0usize, 0.0f64);
        for ((&x, &sv), &bd) in got.iter().zip(&r.score).zip(&r.kernel) {
            let d = (f64::from(x) - sv).abs();
            over += usize::from(outside(d, bd));
            worst = worst.max(d / bd);
        }
        let top = exact_top(got, k);
        let exact = out.list[..k] == top[..];
        let kth = top
            .iter()
            .map(|&i| key_of(got[i as usize]))
            .min()
            .ok_or("an empty selection")?;
        let same_bin = got
            .iter()
            .filter(|&&v| key_of(v) >> 22 == kth >> 22)
            .count();
        let same_prefix = got
            .iter()
            .filter(|&&v| key_of(v) >> 11 == kth >> 11)
            .count();
        let ties = got.iter().filter(|&&v| key_of(v) == kth).count();
        let shape = list_shape(&out.list, k, n);
        let pass = over == 0 && exact && shape && out.hist_zero;
        println!(
            "depth-clustered n={n} top_k={k} rows_in_kth_top10_bin={same_bin} rows_in_kth_top21={same_prefix} \
             rows_equal_kth={ties} rule_over={over}/{n} rule_worst={worst:.3} exact_top={exact} \
             list_shape={shape} hist_zero={} {}",
            out.hist_zero,
            verdict(pass)
        );
        Ok(pass)
    }

    // ------------------------------------------------------------- replay

    /// (vi) One graph captured at 32,768 rows and the file's top_k, replayed
    /// with the counts rewritten in place — deeper and shallower, a narrower
    /// selection, one row, the identity and its boundary — each replay
    /// bit-identical to an eager launch of the same counts, whose list is the
    /// exact top-k of its scores.
    fn replay(cx: &Cx) -> Result<bool, GateError> {
        let stream = cx.gpu.stream();
        let n_max = 32_768usize;
        let k_file = cx.hp.indexer.top_k;
        let stride = k_file + LIST_SLACK;
        let s = synth(cx, n_max, 31)?;
        let nv = [u32::try_from(n_max)?];
        let mut launch = Launch::new(
            stream, &s.q_raw, &s.w_raw, &nv, k_file, &s.cs, &s.keys, stride,
        )?;
        let graph = cx.gpu.capture(|_s| launch.enqueue(&cx.kernels, stream))?;
        let variants = [
            (n_max, k_file),
            (20_001, k_file),
            (n_max, 64),
            (4_097, 1),
            (400, k_file),
            (k_file, k_file),
            (k_file + 1, k_file),
        ];
        let (mut same, mut exact) = (0usize, 0usize);
        for &(n, k) in &variants {
            let nv = [u32::try_from(n)?];
            launch.set_counts(stream, &nv, k)?;
            let want = launch.run(&cx.kernels, stream)?;
            let top_ok = if n > k {
                want.list[..k] == exact_top(&want.scores[..n], k)[..]
                    && list_shape(&want.list, k, n)
            } else {
                want.list[..n]
                    .iter()
                    .enumerate()
                    .all(|(i, &v)| v as usize == i)
                    && want.list[n..].iter().all(|&v| v == LIST_POISON)
            };
            exact += usize::from(top_ok);
            launch.poison(stream)?;
            graph.launch(stream)?;
            let got = launch.read(stream)?;
            same += usize::from(same_out(&want, &got));
        }
        let m = variants.len();
        let pass = same == m && exact == m;
        println!(
            "replay rows={n_max} nodes={} counts={} bit_identical_to_eager={same}/{m} eager_lists_exact={exact}/{m} {}",
            graph.node_count(),
            variants
                .iter()
                .map(|(n, k)| format!("{n}/{k}"))
                .collect::<Vec<_>>()
                .join(","),
            verdict(pass)
        );
        Ok(pass)
    }

    // ------------------------------------------------------- candidate mask

    /// f16's unit roundoff, 2⁻¹¹, and its smallest subnormal step's half,
    /// 2⁻²⁵: `|f16(y) − y| <= 2⁻¹¹·|y| + 2⁻²⁵`.
    const F16_U: f64 = 1.0 / 2048.0;
    const F16_SUB: f64 = 1.0 / 33_554_432.0;

    /// One indexer layer's inputs from the candidate set: ik's projections
    /// (`lid_q`, `lid_weights`) as our score pass takes them, ik's own roped
    /// query and scaled weights, and its index keys of the visible rows.
    struct CandInputs {
        q_raw: Vec<f32>,
        w_raw: Vec<f32>,
        q_roped: Vec<f32>,
        w_scaled: Vec<f32>,
        keys: Vec<u16>,
    }

    fn cand_inputs(man: &RefManifest, l: usize, n_vis: usize) -> Result<CandInputs, GateError> {
        let dir = &man.dir;
        let (_, q_row) = node(man, &format!("lid_q-{l}"), "MUL_MAT")?;
        let q_name = format!("blk.{l}.indexer.attn_q_b.weight");
        if q_row.src0.as_deref() != Some(q_name.as_str()) {
            return Err(format!("lid_q-{l} multiplies {:?}, want {q_name}", q_row.src0).into());
        }
        let (at_w, w_row) = node(man, &format!("lid_weights-{l}"), "MUL_MAT")?;
        let scale_row = man.tensors[at_w..]
            .iter()
            .find(|r| r.op == "SCALE" && r.src0.as_deref() == Some(w_row.name.as_str()))
            .ok_or_else(|| format!("no SCALE of lid_weights-{l}"))?;
        let (_, roped_row) = node(man, &format!("indexer_q-{l}"), "ROPE")?;
        let (_, k_row) = man.tensor_at(&format!("lid_k-{l}"), 0)?;
        if k_row.ty != "f16" || k_row.ne[0] != HEAD_DIM as u64 {
            return Err(format!(
                "lid_k-{l} is {} with rows of {}, want f16 rows of {HEAD_DIM}",
                k_row.ty, k_row.ne[0]
            )
            .into());
        }
        let keys = widened_f16_rows_in(dir, k_row)?;
        if keys.len() < n_vis * HEAD_DIM {
            return Err(format!(
                "lid_k-{l} holds {} rows, the plan sees {n_vis}",
                keys.len() / HEAD_DIM
            )
            .into());
        }
        let inputs = CandInputs {
            q_raw: ref_tensor_logical_in(dir, q_row)?,
            w_raw: ref_tensor_logical_in(dir, w_row)?,
            q_roped: ref_tensor_logical_in(dir, roped_row)?,
            w_scaled: ref_tensor_logical_in(dir, scale_row)?,
            keys: keys[..n_vis * HEAD_DIM].to_vec(),
        };
        let nq = HEADS * HEAD_DIM;
        if [inputs.q_raw.len(), inputs.q_roped.len()] != [nq; 2]
            || [inputs.w_raw.len(), inputs.w_scaled.len()] != [HEADS; 2]
        {
            return Err(
                format!("layer {l}: the query chain is not {HEADS} heads of {HEAD_DIM}").into(),
            );
        }
        Ok(inputs)
    }

    /// The bound on one value of `fast_ht` over `x` against the exact
    /// orthonormal transform: seven butterfly levels and the scale's rounding
    /// (`γ(8)` of `HT_SCALE·Σ|x|`, one more rounding for the f64 reference the
    /// gate compares with), and the f32 scale's distance to `2^-3.5`.
    fn ht_bound(x: &[f32]) -> f64 {
        let sum: f64 = x.iter().map(|&v| f64::from(v).abs()).sum();
        let exact = (HEAD_DIM as f64).sqrt().recip();
        let ds = (f64::from(HT_SCALE) - exact).abs();
        gamma(9) * f64::from(HT_SCALE) * sum + ds * sum
    }

    /// The exact orthonormal transform of `x` in f64: `fast_ht`'s butterflies
    /// with the scale `1/√n`.
    fn ht_exact(x: &[f32]) -> Vec<f64> {
        let mut v: Vec<f64> = x.iter().map(|&a| f64::from(a)).collect();
        let n = v.len();
        let mut h = 1;
        while h < n {
            for i in (0..n).step_by(2 * h) {
                for j in i..i + h {
                    let (a, b) = (v[j], v[j + h]);
                    v[j] = a + b;
                    v[j + h] = a - b;
                }
            }
            h <<= 1;
        }
        let s = (n as f64).sqrt().recip();
        v.iter().map(|a| a * s).collect()
    }

    /// ik's keys in our basis: per row `f16(fast_ht(k))`, the transform our
    /// index-key kernel applies before its f16 store.
    fn keys_in_our_basis(keys: &[u16]) -> Vec<u16> {
        keys.chunks(HEAD_DIM)
            .flat_map(|r| {
                let mut v: Vec<f32> = r.iter().map(|&b| half_to_f32(b)).collect();
                fast_ht(&mut v);
                v.into_iter().map(f32_to_f16_bits).collect::<Vec<_>>()
            })
            .collect()
    }

    /// What the basis change costs a row's score, and ik's exact rule.
    struct Basis {
        /// ik's exact score of the row: `Σ_h w_h · relu(Σ_d q_d·k_d)`, ik's
        /// roped query and scaled weights and its f16 keys, summed in f64.
        exact: Vec<f64>,
        /// ik's distance to `exact` (`Rule::ik`'s form over ik's operands).
        ik: Vec<f64>,
        /// The distance of our f32 query and our-basis keys' exact score to
        /// `exact`: per head `Σ_d bq_d·|k''_d| + (|q_d| + bq_d)·bk_d`, `bq` the
        /// transform's bound on a query value, `bk` the transform's and the
        /// f16 store's on a key value, carried by `|w_h|`.
        basis: Vec<f64>,
        /// Query and key values farther from the exact transform of ik's than
        /// their bound: a basis the gate built wrong.
        over: usize,
    }

    /// [`Basis`] of `n` rows: `q` our kernel's query (after its transform),
    /// `keys` our-basis keys, `inp` ik's operands.
    fn basis(q: &[f32], keys: &[u16], inp: &CandInputs, n: usize) -> Basis {
        let mut over = 0usize;
        let mut bq = vec![0.0f64; HEADS * HEAD_DIM];
        for h in 0..HEADS {
            let x = &inp.q_roped[h * HEAD_DIM..(h + 1) * HEAD_DIM];
            let (exact, b) = (ht_exact(x), ht_bound(x));
            for (d, &e) in exact.iter().enumerate() {
                let i = h * HEAD_DIM + d;
                bq[i] = b;
                over += usize::from(outside((f64::from(q[i]) - e).abs(), b));
            }
        }
        let w: Vec<f64> = inp.w_scaled.iter().map(|&v| f64::from(v)).collect();
        let mut out = Basis {
            exact: Vec::with_capacity(n),
            ik: Vec::with_capacity(n),
            basis: Vec::with_capacity(n),
            over: 0,
        };
        for t in 0..n {
            let kr: Vec<f32> = inp.keys[t * HEAD_DIM..(t + 1) * HEAD_DIM]
                .iter()
                .map(|&b| half_to_f32(b))
                .collect();
            let (kx, kb) = (ht_exact(&kr), ht_bound(&kr));
            let ko: Vec<f64> = keys[t * HEAD_DIM..(t + 1) * HEAD_DIM]
                .iter()
                .map(|&b| f64::from(half_to_f32(b)))
                .collect();
            let bk: Vec<f64> = kx
                .iter()
                .map(|&y| kb + F16_U * (y.abs() + kb) + F16_SUB)
                .collect();
            over += ko
                .iter()
                .zip(&kx)
                .zip(&bk)
                .filter(|&((&a, &e), &b)| outside((a - e).abs(), b))
                .count();
            let (mut ex, mut ikb, mut smag, mut bs) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
            for (h, &wh) in w.iter().enumerate() {
                let (mut dot, mut pq, mut bb) = (0.0f64, 0.0f64, 0.0f64);
                for d in 0..HEAD_DIM {
                    let i = h * HEAD_DIM + d;
                    let qr = f64::from(inp.q_roped[i]);
                    let kv = f64::from(kr[d]);
                    dot += qr * kv;
                    pq += (qr * kv).abs();
                    let qo = f64::from(q[i]).abs();
                    bb += bq[i] * ko[d].abs() + (qo + bq[i]) * bk[d];
                }
                let relu = dot.max(0.0);
                ex += wh * relu;
                smag += wh.abs() * relu;
                ikb += wh.abs() * gamma(HEAD_DIM) * pq;
                bs += wh.abs() * bb;
            }
            out.exact.push(ex);
            out.ik.push(ikb + gamma(HEADS) * smag + U * ex.abs());
            out.basis.push(bs);
        }
        out.over = over;
        out
    }

    /// A list `ids` of rows of `cand` (ascending rows) against the rule's
    /// selection of `k` among them by `score`, outside the tie band
    /// ([`ids_vs_rule`] over the candidate rows): (disagreeing rows, rows in
    /// the band). An id outside `cand` disagrees.
    fn ids_vs_rule_among(
        ids: &[u32],
        cand: &[usize],
        score: &[f64],
        beta: &[f64],
        k: usize,
    ) -> (usize, usize) {
        let sub: Vec<f64> = cand.iter().map(|&r| score[r]).collect();
        let sb: Vec<f64> = cand.iter().map(|&r| beta[r]).collect();
        let mut foreign = 0usize;
        let slots: Vec<u32> = ids
            .iter()
            .filter_map(|&i| match cand.binary_search(&(i as usize)) {
                Ok(s) => u32::try_from(s).ok(),
                Err(_) => {
                    foreign += 1;
                    None
                }
            })
            .collect();
        let (off, band) = ids_vs_rule(&slots, &sub, &sb, k);
        (off + foreign, band)
    }

    /// The rows of blocks `kept` (of `block` rows) below `n`, ascending.
    fn rows_of(kept: &[u32], block: usize, n: usize) -> Vec<usize> {
        let mut r: Vec<usize> = kept
            .iter()
            .flat_map(|&b| {
                let b = b as usize;
                b * block..((b + 1) * block).min(n)
            })
            .collect();
        r.sort_unstable();
        r
    }

    /// A tensor row's integers as distinct ascending `u32`s, refused by name
    /// unless there are `k` of them, distinct, below `n`.
    fn int_set(man: &RefManifest, name: &str, k: usize, n: usize) -> Result<Vec<u32>, GateError> {
        let v: Vec<u32> = ref_ints(man, name, 0, RowKind::Tensor, Layout::Flat)?
            .into_iter()
            .map(u32::try_from)
            .collect::<Result<_, _>>()?;
        let mut s = v.clone();
        s.sort_unstable();
        s.dedup();
        if v.len() != k || s.len() != k || s.iter().any(|&i| i as usize >= n) {
            return Err(format!("{name}: {v:?} is not {k} distinct entries below {n}").into());
        }
        Ok(s)
    }

    fn symdiff(a: &[u32], b: &[u32]) -> usize {
        a.iter().filter(|i| b.binary_search(i).is_err()).count()
            + b.iter().filter(|i| a.binary_search(i).is_err()).count()
    }

    /// (vii) The candidate mask against ik's at `…_d1c_unfused_every_node`.
    fn cand_set(cx: &mut Cx) -> Result<bool, GateError> {
        let dir = CAND.path(D1C);
        let man = RefManifest::open(&dir, &CAND)?;
        let info = set_info(cx, &man)?;
        let flags = man.header.flags.as_deref().unwrap_or("");
        let blocks = kv_override(flags, "attention.candidate_topk_blocks")?.ok_or_else(|| {
            format!(
                "{}: no candidate_topk_blocks override in # flags",
                dir.display()
            )
        })?;
        // The separate graph's marks: its candidate nodes, and no indexer
        // Hadamard (a dump without V41_SEPARATE has the latter, not the former).
        if let Some(r) = man
            .tensors
            .iter()
            .find(|r| r.name.starts_with("lid_q_hadamard-"))
        {
            return Err(format!(
                "{}: holds {} — dumped without ik's separate graph (V41_SEPARATE)",
                dir.display(),
                r.name
            )
            .into());
        }
        let hp = cx
            .hp
            .clone()
            .with_indexer_top_k(info.top_k)
            .with_candidate_topk_blocks(blocks)?;
        let mask = hp
            .candidates
            .ok_or("the model has no candidate mask to hold to ik's")?;
        let (src, block, k) = (mask.source_layer, mask.block_size, info.top_k);
        let consumers: Vec<usize> = (0..hp.layers.len())
            .filter(|&l| hp.candidate_role(l) == Some(CandidateRole::Consumer))
            .collect();
        let (n, rows) = layer_rows(&info, src)?;
        let nb = n.div_ceil(block);
        println!(
            "set cand: {} (build {}) — position {}, top_k {k}, {blocks} blocks of {block}, source layer \
             {src} ({n} visible rows, {nb} blocks), consumers {}",
            man.dir.display(),
            man.build.as_deref().unwrap_or("-"),
            info.pos,
            list(&consumers)
        );
        if nb <= blocks || n <= k {
            return Err(format!(
                "cand: {n} rows in {nb} blocks select nothing at {blocks} blocks, top_k {k}"
            )
            .into());
        }
        let mut ok = true;

        // 1. The rule against ik, on ik's own data.
        let (_, s_row) = node(&man, &format!("lid_score_chunk-{src}"), "ADD")?;
        let ik_scores = raw_f32(&man, s_row)?;
        let (_, bs_row) = node(&man, &format!("cand_block_score-{src}"), "POOL_2D")?;
        let bs = raw_f32(&man, bs_row)?;
        let (_, pin_row) = node(&man, &format!("cand_block_score_pin-{src}"), "ADD")?;
        let bs_pin = raw_f32(&man, pin_row)?;
        let (_, keep_row) = man.tensor_at(&format!("cand_keep-{src}"), 0)?;
        // A view of the whole keep tensor: its plain file is the tensor, −inf kept.
        if keep_row.contig != Some(1) {
            return Err(format!(
                "cand_keep-{src} is not contiguous: its plain file is not the tensor"
            )
            .into());
        }
        let keep = raw_f32(&man, keep_row)?;
        let nb_ik = bs.len();
        if ik_scores.len() != nb_ik * block || nb_ik < nb || bs_pin.len() != nb_ik {
            return Err(format!(
                "cand: ik ranks {nb_ik} blocks of {block} over {} scores, the plan {nb} blocks",
                ik_scores.len()
            )
            .into());
        }
        let ik_kept = int_set(&man, &format!("cand_block_top_k-{src}"), blocks, nb)?;
        // ggml's POOL_MAX: seeded with −f32::MAX, a value taken only when it
        // is larger, so a block of masked rows keeps the seed.
        let pool_max = (0..nb_ik).all(|b| {
            let m = ik_scores[b * block..(b + 1) * block]
                .iter()
                .fold(f32::MIN, |m, &v| if v > m { v } else { m });
            m.to_bits() == bs[b].to_bits()
        });
        let masked = ik_scores[n..].iter().all(|&v| v == f32::NEG_INFINITY)
            && ik_scores[..n].iter().all(|v| v.is_finite());
        let padded = bs[nb..].iter().all(|&v| v == f32::MIN);
        let pin_ok = (0..nb_ik).all(|b| {
            if b == nb - 1 {
                bs_pin[b] == f32::INFINITY
            } else {
                bs_pin[b].to_bits() == bs[b].to_bits()
            }
        });
        let ik_rule = ds41_cand::kept_rule(&bs[..nb], nb, blocks, 1)
            .ok_or("cand: the rule keeps every block")?;
        // Equal keys at the last kept place of the others: ik's top-k breaks
        // their tie in an order of its own.
        let others: Vec<usize> = (0..nb - 1).collect();
        let mut order = others.clone();
        order.sort_by(|&a, &b| {
            ds41_cand::key_of(bs[b])
                .cmp(&ds41_cand::key_of(bs[a]))
                .then(a.cmp(&b))
        });
        let last = ds41_cand::key_of(bs[order[blocks - 2]]);
        let tied: Vec<u32> = if ik_rule.tie {
            others
                .iter()
                .filter(|&&b| ds41_cand::key_of(bs[b]) == last)
                .map(|&b| u32::try_from(b).unwrap_or(u32::MAX))
                .collect()
        } else {
            Vec::new()
        };
        let rule_off = (0..nb)
            .map(|b| u32::try_from(b).unwrap_or(u32::MAX))
            .filter(|b| tied.binary_search(b).is_err())
            .filter(|b| ik_kept.binary_search(b).is_ok() != ik_rule.kept.binary_search(b).is_ok())
            .count();
        let keep_ok = keep.len() == nb_ik
            && keep.iter().enumerate().all(|(b, &v)| {
                let kept = u32::try_from(b).is_ok_and(|b| ik_kept.binary_search(&b).is_ok());
                if kept {
                    v == 0.0
                } else {
                    v == f32::NEG_INFINITY
                }
            });
        let pass1 = pool_max && masked && padded && pin_ok && rule_off == 0 && keep_ok;
        ok &= pass1;
        println!(
            "cand 1 rule-vs-ik layer={src} blocks={nb}/{nb_ik} kept={blocks} ik_kept={} pool_max_of_scores={pool_max} \
             rows_from_n_masked={masked} padded_blocks_pool_seed={padded} pin_on_newest={pin_ok} \
             rule_vs_ik_off={rule_off} tie_band_blocks={} keep_pattern={keep_ok} {}",
            list(&ik_kept),
            tied.len(),
            verdict(pass1)
        );

        // 2. Our kernels on ik's inputs of the source layer.
        let stream = cx.gpu.stream();
        cx.gpu.clear_fault()?;
        let (cs, cs_ik) = cx.tables(info.pos)?;
        let tables_same = bits_equal(&cs, &cs_ik);
        let stride = k + LIST_SLACK;
        let nv = [u32::try_from(n)?];
        let shape = CandShape::new(blocks, block)?;
        let launch_keys = |inp: &CandInputs| {
            let mut keys = vec![NAN_F16; rows * HEAD_DIM];
            keys[..n * HEAD_DIM].copy_from_slice(&keys_in_our_basis(&inp.keys));
            keys
        };
        let si = cand_inputs(&man, src, n)?;
        let src_keys = launch_keys(&si);
        let mut sl = Launch::new(stream, &si.q_raw, &si.w_raw, &nv, k, &cs, &src_keys, stride)?;
        sl.poison(stream)?;
        sl.enqueue(&cx.kernels, stream)?;
        let mut scratch = CandScratch::new(stream, shape, 1, rows)?;
        let mut kept_dev = DeviceBuffer::from_host(stream, &vec![LIST_POISON; blocks])?;
        cx.cand.enqueue_select(
            stream,
            SelectArgs {
                ints: &sl.ints,
                n_at: N_VIS_AT,
                scores: &sl.scratch.scores,
                score_k: k,
                rows,
                tokens: 1,
                shape,
                fault: cx.gpu.unlabelled_sink(),
                scratch: &mut scratch,
                kept: &mut kept_dev,
                kstride: blocks,
            },
        )?;
        let so = sl.read(stream)?;
        let ours_kept = kept_dev.to_host_vec(stream)?;
        let q_is_ik = bits_equal(&so.q[..HEADS * HEAD_DIM], &query(&si.q_roped, &[]))
            && bits_equal(&so.w[..HEADS], &si.w_scaled);
        let sr = rule(&so.q[..HEADS * HEAD_DIM], &so.w[..HEADS], &src_keys, n);
        let sb = basis(&so.q[..HEADS * HEAD_DIM], &src_keys, &si, n);
        let (mut ours_over, mut ik_over, mut dump_over) = (0usize, 0usize, 0usize);
        let beta: Vec<f64> = (0..n)
            .map(|t| sr.kernel[t] + sr.rep[t] + sb.basis[t] + sb.ik[t])
            .collect();
        for t in 0..n {
            let (g, s) = (f64::from(so.scores[t]), f64::from(ik_scores[t]));
            ours_over += usize::from(outside(
                (g - sb.exact[t]).abs(),
                sr.kernel[t] + sr.rep[t] + sb.basis[t],
            ));
            ik_over += usize::from(outside((s - sb.exact[t]).abs(), sb.ik[t]));
            dump_over += usize::from(outside((g - s).abs(), beta[t]));
        }
        let past = so.scores[n..]
            .iter()
            .all(|v| v.to_bits() == POISON.to_bits());
        let own_rule = ds41_cand::kept_rule(&so.scores[..n], n, blocks, block)
            .is_some_and(|r| r.kept == ours_kept);
        // The rule's kept blocks by ik's exact scores, each block's key its
        // largest row and its band its rows' largest: the pin and the
        // `blocks − 1` best others, outside the band of the last kept key.
        let bkey: Vec<f64> = (0..nb)
            .map(|b| {
                (b * block..((b + 1) * block).min(n))
                    .map(|t| sb.exact[t])
                    .fold(f64::NEG_INFINITY, f64::max)
            })
            .collect();
        let bbeta: Vec<f64> = (0..nb)
            .map(|b| {
                (b * block..((b + 1) * block).min(n))
                    .map(|t| beta[t])
                    .fold(0.0, f64::max)
            })
            .collect();
        let pin = u32::try_from(nb - 1)?;
        let others_of =
            |kept: &[u32]| -> Vec<u32> { kept.iter().copied().filter(|&b| b != pin).collect() };
        let (dev_off, block_band) = ids_vs_rule(
            &others_of(&ours_kept),
            &bkey[..nb - 1],
            &bbeta[..nb - 1],
            blocks - 1,
        );
        let (ik_off, _) = ids_vs_rule(
            &others_of(&ik_kept),
            &bkey[..nb - 1],
            &bbeta[..nb - 1],
            blocks - 1,
        );
        let ours_shape =
            ours_kept.windows(2).all(|p| p[0] < p[1]) && ours_kept.last() == Some(&pin);
        let kept_symdiff = symdiff(&ours_kept, &ik_kept);
        let pass2 = q_is_ik
            && tables_same
            && sb.over == 0
            && ours_over == 0
            && ik_over == 0
            && dump_over == 0
            && past
            && own_rule
            && ours_shape
            && dev_off == 0
            && ik_off == 0
            && so.hist_zero;
        ok &= pass2;
        println!(
            "cand 2 kernels-on-ik layer={src} n_vis={n} rows={rows} kept={} q_w_are_ik={q_is_ik} \
             table_is_ggml={tables_same} basis_over={} ours_vs_exact_over={ours_over}/{n} \
             ik_vs_exact_over={ik_over}/{n} ours_vs_ik_over={dump_over}/{n} past_n_vis_untouched={past} \
             kept_is_rule_of_ours={own_rule} kept_shape={ours_shape} ours_vs_rule_off={dev_off} \
             ik_vs_rule_off={ik_off} tie_band_blocks={block_band} ours_vs_ik_symdiff={kept_symdiff} \
             hist_zero={} {}",
            list(&ours_kept),
            sb.over,
            so.hist_zero,
            verdict(pass2)
        );

        // 3. The consumers, over our kept blocks.
        let mut counts = DeviceBuffer::from_host(stream, &[LIST_POISON; 2])?;
        let mut changed_total = 0usize;
        for &l in &consumers {
            let (nl, rl) = layer_rows(&info, l)?;
            if (nl, rl) != (n, rows) {
                return Err(format!(
                    "cand: consumer {l} sees {nl} of {rl} rows, the source {n} of {rows}"
                )
                .into());
            }
            let ci = cand_inputs(&man, l, n)?;
            let keys = launch_keys(&ci);
            let mut cl = Launch::new(stream, &ci.q_raw, &ci.w_raw, &nv, k, &cs, &keys, stride)?;
            cl.poison(stream)?;
            cx.kernels.enqueue_score(stream, &mut cl.args())?;
            stream.synchronize()?;
            let unmasked = cl.scratch.scores.to_host_vec(stream)?;
            cx.cand.enqueue_compact(
                stream,
                CompactArgs {
                    ints: &cl.ints,
                    n_at: N_VIS_AT,
                    top_k_at: N_VIS_AT + 1,
                    kept: &kept_dev,
                    kstride: blocks,
                    rows,
                    tokens: 1,
                    stride,
                    shape,
                    fault: cx.gpu.unlabelled_sink(),
                    scores: &mut cl.scratch.scores,
                    hist: &mut cl.scratch.hist,
                    counts: &mut counts,
                    counts_at: 0,
                },
            )?;
            stream.synchronize()?;
            let compacted = cl.scratch.scores.to_host_vec(stream)?;
            cx.kernels
                .enqueue_topk(stream, &mut cl.args(), &counts, 0, 1)?;
            cx.cand.enqueue_remap(
                stream,
                RemapArgs {
                    ints: &cl.ints,
                    n_at: N_VIS_AT,
                    kept: &kept_dev,
                    kstride: blocks,
                    counts: &counts,
                    counts_at: 0,
                    rows,
                    tokens: 1,
                    shape,
                    fault: cx.gpu.unlabelled_sink(),
                    list: &mut cl.list,
                    stride,
                },
            )?;
            let co = cl.read(stream)?;
            let row = ds41_cand::check(&ds41_cand::Card {
                n,
                top_k: k,
                blocks,
                block,
                source: &so.scores[..n],
                kept: &ours_kept,
                unmasked: &unmasked,
                compacted: &compacted,
                list: &co.list,
            });
            changed_total += row.changed;
            let list_shape_ok = co.list[k..].iter().all(|&i| i == LIST_POISON);
            // Against ik: its list among its own kept blocks' rows, ours among
            // ours, each outside the tie band of the k-th candidate row.
            let cr = rule(&co.q[..HEADS * HEAD_DIM], &co.w[..HEADS], &keys, n);
            let cb = basis(&co.q[..HEADS * HEAD_DIM], &keys, &ci, n);
            let cbeta: Vec<f64> = (0..n)
                .map(|t| cr.kernel[t] + cr.rep[t] + cb.basis[t] + cb.ik[t])
                .collect();
            let ours_over = (0..n)
                .filter(|&t| {
                    outside(
                        (f64::from(unmasked[t]) - cb.exact[t]).abs(),
                        cr.kernel[t] + cr.rep[t] + cb.basis[t],
                    )
                })
                .count();
            let ik_ids = int_set(&man, &format!("lid_top_k-{l}"), k, n)?;
            let ours_ids: Vec<u32> = co.list[..k].to_vec();
            let (dev_off, band) = ids_vs_rule_among(
                &ours_ids,
                &rows_of(&ours_kept, block, n),
                &cb.exact,
                &cbeta,
                k,
            );
            let (ik_off, _) =
                ids_vs_rule_among(&ik_ids, &rows_of(&ik_kept, block, n), &cb.exact, &cbeta, k);
            let q_is_ik = bits_equal(&co.q[..HEADS * HEAD_DIM], &query(&ci.q_roped, &[]))
                && bits_equal(&co.w[..HEADS], &ci.w_scaled);
            // ik's own mask: 0 on its kept blocks' visible rows, −inf elsewhere.
            let (_, m_row) = node(&man, &format!("lid_mask_cand-{l}"), "ADD")?;
            let ik_mask = raw_f32(&man, m_row)?;
            let ik_rows = rows_of(&ik_kept, block, n);
            let ik_mask_ok = ik_mask.len() >= n
                && ik_mask.iter().enumerate().all(|(t, &v)| {
                    if ik_rows.binary_search(&t).is_ok() {
                        v == 0.0
                    } else {
                        v == f32::NEG_INFINITY
                    }
                });
            // ik's scores of the layer (the unfused set's node): its exact rule
            // within its band on its kept blocks' rows, −inf on every other row,
            // and ours within both bands of them.
            let (_, cs_row) = node(&man, &format!("lid_score_chunk-{l}"), "ADD")?;
            let ik_sc = raw_f32(&man, cs_row)?;
            let (mut ik_over, mut ours_vs_ik) = (0usize, 0usize);
            let mut ik_hidden = ik_sc.len() >= n;
            for (t, &v) in ik_sc.iter().enumerate() {
                if t < n && ik_rows.binary_search(&t).is_ok() {
                    let e = cb.exact[t];
                    ik_over += usize::from(outside((f64::from(v) - e).abs(), cb.ik[t]));
                    ours_vs_ik += usize::from(outside(
                        (f64::from(unmasked[t]) - f64::from(v)).abs(),
                        cbeta[t],
                    ));
                } else {
                    ik_hidden &= v == f32::NEG_INFINITY;
                }
            }
            let pass = row.pass()
                && row.selects
                && list_shape_ok
                && q_is_ik
                && ik_over == 0
                && ours_vs_ik == 0
                && ik_hidden
                && cb.over == 0
                && ours_over == 0
                && dev_off == 0
                && ik_off == 0
                && ik_mask_ok
                && co.hist_zero;
            ok &= pass;
            println!(
                "cand 3 consumer layer={l} {} | q_w_are_ik={q_is_ik} basis_over={} \
                 ours_vs_exact_over={ours_over}/{n} ours_vs_rule_off={dev_off} ik_vs_rule_off={ik_off} \
                 ik_vs_exact_over={ik_over}/{} ours_vs_ik_over={ours_vs_ik} ik_hides_the_rest={ik_hidden} \
                 tie_band_rows={band} ours_vs_ik_symdiff={} ik_mask_is_its_kept={ik_mask_ok} \
                 list_rest_untouched={list_shape_ok} hist_zero={} {}",
                row.line(),
                cb.over,
                ik_rows.len(),
                symdiff(&ours_ids, &ik_ids),
                co.hist_zero,
                verdict(pass)
            );
        }
        let witness = changed_total > 0;
        let fault = cx.gpu.fault()?;
        let clean = fault.is_none();
        ok &= witness && clean;
        println!(
            "cand witness: entries the mask changed over {} consumers {changed_total} (a consumer that \
             ignores the mask is visible only when > 0); fault word {fault:?} — {}",
            consumers.len(),
            verdict(witness && clean)
        );
        Ok(ok)
    }
}
