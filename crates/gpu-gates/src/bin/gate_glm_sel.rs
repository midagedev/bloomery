//! GPU gate for GLM-5.3-Flash's k-pool selector on the card: the pool pass
//! (`latent::index_pool`), the score pass (`kpool::kpool_score`), the top-k
//! with the higher pool of a tie (`qsa::qsa_topk_high`), and the V4.1
//! attention reading the list it writes (`ds41_attn_seg_sel`). Clauses 1–6
//! run on synthetic inputs from fixed seeds at the file's shapes (32 indexer
//! heads of 128, pools of 4, 512 pools kept, lists of 2,051); clauses 7–10
//! on ik's `--dsa` step sets (refset family `ik-glm5next-dsa`), every latent
//! layer teacher-forced on ik's own inputs at the step.
//!
//! Clauses:
//! 1. `pool`: the pool pass over every count `1 ..= 2,048` of a random index
//!    cache in one launch, into a NaN plane: every bit of the plane the host
//!    rule's (`index_pool_host`) — the completed pools written, the rest
//!    untouched — and no fault. Each rule mutant (no position bias, the bias
//!    transposed, the plain mean of the members, the pool one row over)
//!    must move at least one bit.
//! 2. `score`: the score pass over a 4,096-row plane for four tokens at
//!    counts 2,051 (sees 512 pools: not scored), 2,052, 9,001 and 16,384,
//!    against the rule (the kernel's split query, its weights, the exact f16
//!    products summed in f64) within the tensor-core band per pool
//!    (`gate_deepseek41_index`'s derivation, the same arithmetic); every
//!    slot past a row's pools, and every slot of the unscored row, left NaN;
//!    each token alone bit for bit its column of the four-token launch; a
//!    rerun bit for bit. Each rule mutant (the weights' scale `1/√128`,
//!    `relu` dropped, head `h` weighted by head `h + 1`) must sit past the
//!    band at some pool.
//! 3. `topk`: the top-k over clause 2's scores and a planted row of ties at
//!    the cut (relu zeros of both signs and a duplicated positive score),
//!    against `runtime::qsa::select_by(.., Tie::Higher)`: every list and its
//!    length (the second visible-count word) exactly, the slots past it
//!    untouched; counts up to 2,051 give `0 .. c`; a refused count (0, past
//!    the cache) gives length 0 and [`FaultSite::PoolSelect`]. Mutants: the
//!    lower pool of a tie, 513 pools kept, the tail dropped, the identity
//!    from 1 — each must differ from the card's lists.
//! 4. `identity`: at counts 1, 64, 65, 1,000 and 2,051 the attention over the
//!    list the top-k wrote (a stride of 2,051: 33 segments) equals the prefix
//!    attention over a 4,096-row cache (64 segments) bit for bit: the live
//!    segments walk the same rows in the same order, and a neutral segment
//!    is skipped by the merge, not folded. The list shifted by one position
//!    must not.
//! 5. `fault`: a NaN pool member (the pool pass), a NaN query (the score
//!    pass), a refused count (the top-k), a count past the cache (the pool
//!    pass) — each raises its site in its layer alone, clean before and
//!    after a clean launch.
//! 6. `shape`: the three new entries compile with no local depot.
//!
//! Per `--dsa` step set and latent layer `L`, at the step's position `q`,
//! `V = (q + 1) / 4` pools seen:
//! 7. `ik pool` (C1): our pool pass over ik's own members (`dsa_pool_k-L`,
//!    `dsa_pool_g-L`, f16 values of its cache) and the file's
//!    `indexer_compressor_ape` against `dsa_indexer_k_pooled-L`: each value
//!    within half an f16 ulp of ik's value and our f32 rule's distance to
//!    ik's (`POOL_UNITS` units of the members' largest magnitude); the
//!    mutants no bias, bias transposed and plain mean past it.
//! 8. `ik score` (C2): our score pass on ik's `dsa_indexer_q-L`,
//!    `dsa_indexer_weights-L` (scale 1: ik's are scaled) and ik's pooled keys
//!    rounded to f16 against `dsa_indexer_score-L` within the kernel's band,
//!    the split's, the f16 rounding of the pools (`Σ_h |w_h| Σ_d |q_d|·|f16(p)
//!    − p|`) and ik's own band.
//! 9. `ik select` (C3): our top-k on ik's scores lists exactly the cells ik
//!    attends (`dsa_top_k-L`, 512 pools and the tail's three real cells:
//!    `(q + 1) % 4 == 3` on every set); on our own scores (clause 8) every
//!    pool the two lists differ by scores within twice clause 8's band of
//!    ik's cut.
//! 10. `ik attn` (C5): our attention over ik's latent cache (`kv_cache-L`),
//!    ik's absorbed queries (`Qcur-L`) and ik's list, against the rule (the
//!    f16 query, the exact dot rounded to f32 and scaled, the softmax and
//!    value sum in f64) within `gate_glm_mla`'s band; ik's
//!    `kqv_compressed-L` is printed against the rule and against ours, not
//!    held (its CPU flash rounds past that band, as `gate_p5` prints it);
//!    the mutant that attends every position up to the step's must sit past
//!    twice the band from the rule on one layer of the set at least.
//!
//! A set that is not a `--dsa` dump (no `--dsa` on the dumper's command
//! line, or no `dsa_indexer_score-3` row) is refused by name: the family's
//! file, build and architecture checks cannot tell a dense set from one.

#[cfg(not(feature = "deepseek41"))]
fn main() {
    eprintln!("gate_glm_sel: built without the `deepseek41` feature; see `just gate-gpu-glm-sel`.");
    std::process::exit(2);
}

#[cfg(feature = "deepseek41")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_glm_sel", gate::run())
}

#[cfg(feature = "deepseek41")]
mod gate {
    use bloomery_gpu::kpool::{DIM, HEADS, KpoolKernels, ScoreArgs, weights_scale};
    use bloomery_gpu::latent::{
        INDEX_HEAD, INDEX_ROW, IndexPoolArgs, LATENT, LatentKernels, POOL, index_pool_host,
        pools_for,
    };
    use bloomery_gpu::qsa::{QsaKernels, TopkHighArgs, list_width};
    use bloomery_gpu::{DeviceTensor, Fault, FaultSink, FaultSite, Gpu};
    use bloomery_gpu_deepseek41::attn::{self, AttnArgs, AttnKernels, SelectedRows};
    use bloomery_gpu_gates::{
        GateError, NAN_F16, RefManifest, bits_equal, checks_failed, data_dir, max_rel_err,
        no_local_depot, open_split, ref_tensor_logical_in, split_f32, topk_ids_logical_within,
        verdict, widened_f16_rows_in,
    };
    use cuda_core::{CudaStream, DeviceBuffer};
    use gguf::Split;
    use gguf::quant::{f32_to_f16_bits, half_to_f32};
    use model::arch::Arch;
    use model::arch::glm5next::names;
    use refset::arch::glm5next::{D3K_DSA, D16K_DSA, IK_DSA};
    use refset::ik::{Layout, plain_is_logical, ref_ints, ref_tensor_logical_masked_in};
    use runtime::qsa::{Qsa, Tie, select_by};

    /// Pools a row keeps, and its longest list.
    const KEPT: usize = 512;
    const WIDTH: usize = list_width(KEPT);
    const _: () = assert!(WIDTH == 2051 && DIM == INDEX_HEAD);
    /// The synthetic score and top-k cases' cache height.
    const CTX: usize = 16_384;
    /// The latent layers (`l % 4 == 3` of 45).
    const LAYERS: [usize; 11] = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43];
    /// The attention's query heads and scale `1/√key_length_mla`.
    const ATT_HEADS: usize = 64;
    const ATT_SCALE: f32 = 0.0625;
    /// f32's unit roundoff, 2⁻²⁴.
    const U: f64 = f32::EPSILON as f64 / 2.0;
    /// The tensor-core dot's bound in units of `U` times its products'
    /// magnitude [derived, `gate_deepseek41_index`'s `TC_UNITS`]: one k-step
    /// errs by at most 36u of its addends' magnitude, eight k-steps a head.
    const TC_UNITS: f64 = 288.0;
    /// Roundings one head's weighted term meets in the kernel's head sum: a
    /// product and three fused multiply-adds along its lane group, then three
    /// butterfly levels.
    const HEAD_SUM_ROUNDINGS: usize = 7;
    /// Our pool rule's distance to ik's before our f16 rounding, in units of
    /// `U` times the members' largest key magnitude [derived]: the members'
    /// softmax weights differ by the two exponentials (`expf_ik` and ik's
    /// CPU `expf`, each within 2 ulps: 4u), the three adds of the sum, its
    /// reciprocal and the weight's product (5u), then the products and the
    /// three adds of the pool (4u) — 13u of `Σ_i w_i |k_i| <= max_i |k_i|`;
    /// 32 holds it with the margin the other gates' bounds take.
    const POOL_UNITS: f64 = 32.0;
    /// What the lists are poisoned with.
    const LIST_POISON: u32 = u32::MAX;

    /// `γ(n) = n·u / (1 − n·u)`.
    fn gamma(n: usize) -> f64 {
        let nu = n as f64 * U;
        nu / (1.0 - nu)
    }

    /// A 64-bit LCG (Knuth's MMIX constants); the high 24 bits as a unit
    /// float.
    struct Lcg(u64);

    impl Lcg {
        fn unit(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.0 >> 40) as f32 / (1u64 << 24) as f32
        }
        fn fill(&mut self, n: usize, lo: f32, hi: f32) -> Vec<f32> {
            (0..n).map(|_| lo + (hi - lo) * self.unit()).collect()
        }
        fn f16(&mut self, n: usize, lo: f32, hi: f32) -> Vec<u16> {
            self.fill(n, lo, hi)
                .into_iter()
                .map(f32_to_f16_bits)
                .collect()
        }
    }

    fn diff_u16(a: &[u16], b: &[u16]) -> usize {
        a.iter().zip(b).filter(|(x, y)| x != y).count() + a.len().abs_diff(b.len())
    }

    fn fault_text(f: Option<Fault>) -> String {
        f.map_or_else(|| "none".to_owned(), |f| f.to_string())
    }

    struct Ctx<'a> {
        gpu: &'a Gpu,
        lk: LatentKernels,
        kk: KpoolKernels,
        qk: QsaKernels,
        ak: AttnKernels,
    }

    impl Ctx<'_> {
        fn stream(&self) -> &CudaStream {
            self.gpu.stream()
        }

        /// The pool pass over `counts` into a copy of `plane`, read back.
        fn pool(
            &self,
            sink: FaultSink,
            cache: &[u16],
            ape: &[f32],
            counts: &[u32],
            plane: &[u16],
        ) -> Result<Vec<u16>, GateError> {
            let s = self.stream();
            let cache = DeviceTensor::upload(s, cache, cache.len() / INDEX_ROW, INDEX_ROW)?;
            let ape = DeviceBuffer::from_host(s, ape)?;
            let n_keys = DeviceBuffer::from_host(s, counts)?;
            let mut pooled = DeviceTensor::upload(s, plane, plane.len() / INDEX_HEAD, INDEX_HEAD)?;
            self.lk.enqueue_index_pool(
                s,
                IndexPoolArgs {
                    cache: &cache,
                    ape: &ape,
                    n_keys: &n_keys,
                    m: counts.len(),
                    fault: sink,
                    pooled: &mut pooled,
                },
            )?;
            s.synchronize()?;
            Ok(pooled.buf().to_host_vec(s)?)
        }

        /// The score pass for `counts.len()` tokens over `plane` (a cache of
        /// `ctx` positions) into NaN scores, read back.
        #[allow(
            clippy::too_many_arguments,
            reason = "the launch's own inputs, each a separate buffer the clauses vary"
        )]
        fn score(
            &self,
            sink: FaultSink,
            q: &[f32],
            w: &[f32],
            counts: &[u32],
            plane: &[u16],
            ctx: usize,
            scale: f32,
        ) -> Result<Vec<f32>, GateError> {
            let s = self.stream();
            let pools = pools_for(ctx);
            let qb = DeviceBuffer::from_host(s, q)?;
            let wb = DeviceBuffer::from_host(s, w)?;
            let n_keys = DeviceBuffer::from_host(s, counts)?;
            let pooled = DeviceTensor::upload(s, plane, plane.len() / DIM, DIM)?;
            let mut scores = DeviceBuffer::from_host(s, &vec![f32::NAN; counts.len() * pools])?;
            self.kk.enqueue_score(
                s,
                ScoreArgs {
                    q: &qb,
                    w: &wb,
                    n_keys: &n_keys,
                    pooled: &pooled,
                    tokens: counts.len(),
                    ctx,
                    kept: KEPT,
                    scale,
                    fault: sink,
                    scores: &mut scores,
                },
            )?;
            s.synchronize()?;
            Ok(scores.to_host_vec(s)?)
        }

        /// The top-k for `counts.len()` rows of `scores` (a cache of `ctx`
        /// positions) into poisoned lists: each row's list and length.
        fn topk(
            &self,
            sink: FaultSink,
            counts: &[u32],
            scores: &[f32],
            ctx: usize,
        ) -> Result<(Vec<u32>, Vec<u32>), GateError> {
            let s = self.stream();
            let m = counts.len();
            let n_keys = DeviceBuffer::from_host(s, counts)?;
            let sc = DeviceBuffer::from_host(s, scores)?;
            let mut list = DeviceBuffer::from_host(s, &vec![LIST_POISON; m * WIDTH])?;
            let mut vis = DeviceBuffer::from_host(s, &vec![LIST_POISON; 2 * m])?;
            self.qk.enqueue_topk_high(
                s,
                TopkHighArgs {
                    n_keys: &n_keys,
                    scores: &sc,
                    ctx,
                    kept: KEPT,
                    m,
                    fault: sink,
                    list: &mut list,
                    vis: &mut vis,
                },
            )?;
            s.synchronize()?;
            let vis = vis.to_host_vec(s)?;
            Ok((
                list.to_host_vec(s)?,
                (0..m).map(|t| vis[2 * t + 1]).collect(),
            ))
        }

        /// The attention of `q` (`tokens · ATT_HEADS` rows of [`LATENT`])
        /// over `cache` (`[rows][LATENT]` f16): the prefix of `counts[t]`
        /// rows, or with `list` (stride [`WIDTH`]) the first `counts[t]`
        /// entries of token `t`'s list. NaN partials and output.
        fn attend(
            &self,
            q: &[f32],
            cache: &[u16],
            counts: &[u32],
            list: Option<&[u32]>,
        ) -> Result<Vec<f32>, GateError> {
            let s = self.stream();
            let tokens = counts.len();
            let rows = cache.len() / LATENT;
            let q_rows = tokens * ATT_HEADS;
            let segs = attn::segments(0, if list.is_some() { WIDTH } else { rows });
            let qb = DeviceBuffer::from_host(s, q)?;
            let cache = DeviceTensor::upload(s, cache, rows, LATENT)?;
            let vis: Vec<u32> = counts.iter().flat_map(|&c| [0, c]).collect();
            let vis = DeviceBuffer::from_host(s, &vis)?;
            let sinks = DeviceBuffer::from_host(s, &[f32::NEG_INFINITY; ATT_HEADS])?;
            let mut part_v =
                DeviceBuffer::from_host(s, &vec![f32::NAN; attn::partials_v_len(q_rows, segs)])?;
            let mut part_ms =
                DeviceBuffer::from_host(s, &vec![f32::NAN; attn::partials_ms_len(q_rows, segs)])?;
            let mut y = DeviceBuffer::from_host(s, &vec![f32::NAN; q_rows * LATENT])?;
            let lb = list.map(|l| DeviceBuffer::from_host(s, l)).transpose()?;
            // SAFETY: zero u16 at the address of the cache's own live
            // allocation, aligned for u16; the view is released below, before
            // the cache can drop.
            let window = unsafe {
                DeviceTensor::<u16>::window(
                    cache.buf().cu_deviceptr(),
                    0,
                    LATENT,
                    self.gpu.context(),
                )
            };
            let r = self.ak.enqueue(
                s,
                AttnArgs {
                    q: &qb,
                    window: &window,
                    compressed: Some(&cache),
                    selected: lb.as_ref().map(|rows| SelectedRows {
                        rows,
                        stride: WIDTH,
                    }),
                    vis: &vis,
                    sinks: &sinks,
                    scale: ATT_SCALE,
                    tokens,
                    heads: ATT_HEADS,
                    part_v: &mut part_v,
                    part_ms: &mut part_ms,
                    y: &mut y,
                    fault: self.gpu.unlabelled_sink(),
                },
            );
            DeviceTensor::release(window);
            r?;
            s.synchronize()?;
            Ok(y.to_host_vec(s)?)
        }
    }

    pub fn run() -> Result<(), GateError> {
        let gpu = Gpu::new()?;
        let ctx = gpu.context();
        let cx = Ctx {
            gpu: &gpu,
            lk: LatentKernels::load(ctx)?,
            kk: KpoolKernels::load(ctx)?,
            qk: QsaKernels::load(ctx)?,
            ak: AttnKernels::load(ctx)?,
        };
        println!(
            "gate_glm_sel: device {} — indexer heads={HEADS} dim={DIM} pool={POOL} kept={KEPT} \
             width={WIDTH}",
            gpu.device_name()?
        );
        gpu.clear_fault()?;
        let mut failed = 0u32;
        let mut clauses = 0u32;
        let mut tally = |pass: bool| {
            clauses += 1;
            failed += u32::from(!pass);
        };
        tally(pool_case(&cx)?);
        let (score_ok, sc) = score_case(&cx)?;
        tally(score_ok);
        tally(topk_case(&cx, &sc)?);
        tally(identity_case(&cx)?);
        for pass in fault_cases(&cx)? {
            tally(pass);
        }
        tally(no_local_depot(&[
            "index_pool",
            "kpool_score",
            "qsa_topk_high",
        ])?);
        let split = open_split(Arch::Glm5next, "just gate-gpu-glm-sel")?;
        for set in [D3K_DSA, D16K_DSA] {
            for pass in ik_set(&cx, &split, set)? {
                tally(pass);
            }
        }
        let pass = failed == 0;
        println!(
            "gate_glm_sel: {clauses} clauses, {failed} failed — {}",
            verdict(pass)
        );
        if !pass {
            return Err(checks_failed());
        }
        Ok(())
    }

    // ------------------------------------------------------------ 1. pool

    /// A pool-rule mutant: the rule as `index_pool_host` has it, with one
    /// change.
    #[derive(Clone, Copy)]
    enum PoolMut {
        NoApe,
        ApeTransposed,
        PlainMean,
        NextPool,
    }

    /// Pool `j`'s key under mutant `mu`, in f32, from `cache` and `ape`.
    fn pool_mutant(cache: &[u16], ape: &[f32], j: usize, mu: PoolMut) -> Vec<f32> {
        let src = if matches!(mu, PoolMut::NextPool) {
            j + 1
        } else {
            j
        };
        (0..INDEX_HEAD)
            .map(|d| {
                let row = |i: usize| &cache[(POOL * src + i) * INDEX_ROW..][..INDEX_ROW];
                let k: Vec<f64> = (0..POOL)
                    .map(|i| f64::from(half_to_f32(row(i)[d])))
                    .collect();
                if matches!(mu, PoolMut::PlainMean) {
                    return (k.iter().sum::<f64>() / POOL as f64) as f32;
                }
                let z: Vec<f64> = (0..POOL)
                    .map(|i| {
                        let a = match mu {
                            PoolMut::NoApe => 0.0,
                            PoolMut::ApeTransposed => ape[(d * POOL + i) % (POOL * INDEX_HEAD)],
                            _ => ape[i * INDEX_HEAD + d],
                        };
                        f64::from(half_to_f32(row(i)[INDEX_HEAD + d])) + f64::from(a)
                    })
                    .collect();
                let mx = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let e: Vec<f64> = z.iter().map(|v| (v - mx).exp()).collect();
                let s: f64 = e.iter().sum();
                (e.iter().zip(&k).map(|(e, k)| e * k).sum::<f64>() / s) as f32
            })
            .collect()
    }

    /// Clause 1.
    fn pool_case(cx: &Ctx<'_>) -> Result<bool, GateError> {
        let rows = 2048usize;
        let mut r = Lcg(0x706f_6f6c);
        let mut cache = Vec::with_capacity(rows * INDEX_ROW);
        for _ in 0..rows {
            cache.extend(r.f16(INDEX_HEAD, -3.0, 3.0));
            cache.extend(r.f16(INDEX_HEAD, -6.0, 6.0));
        }
        let ape = r.fill(POOL * INDEX_HEAD, -1.0, 1.0);
        let counts: Vec<u32> = (1..=rows as u32).collect();
        let plane = vec![NAN_F16; pools_for(rows) * INDEX_HEAD];
        let card = cx.pool(cx.gpu.unlabelled_sink(), &cache, &ape, &counts, &plane)?;
        let word = cx.gpu.take_fault()?;
        let mut want = plane.clone();
        let sites = index_pool_host(&cache, &ape, &counts, &mut want);
        let differ = diff_u16(&card, &want);
        let pools = rows / POOL;
        let mut killed = true;
        let moved: Vec<String> = [
            ("no_ape", PoolMut::NoApe),
            ("ape_transposed", PoolMut::ApeTransposed),
            ("plain_mean", PoolMut::PlainMean),
            ("next_pool", PoolMut::NextPool),
        ]
        .iter()
        .map(|&(name, mu)| {
            let last = if matches!(mu, PoolMut::NextPool) {
                pools - 1
            } else {
                pools
            };
            let n: usize = (0..last)
                .map(|j| {
                    let m = pool_mutant(&cache, &ape, j, mu);
                    (0..INDEX_HEAD)
                        .filter(|&d| f32_to_f16_bits(m[d]) != card[j * INDEX_HEAD + d])
                        .count()
                })
                .sum();
            killed &= n > 0;
            format!("{name}={n}")
        })
        .collect();
        let pass = differ == 0 && sites == 0 && word.is_none() && killed;
        println!(
            "pool counts 1..={rows} in one launch: plane bits differ from the host rule {differ}/{} \
             host_sites={sites:#x} fault={} mutants(moved bits, each > 0): {} {}",
            card.len(),
            fault_text(word),
            moved.join(" "),
            verdict(pass)
        );
        Ok(pass)
    }

    // ----------------------------------------------------------- 2. score

    /// The split of one query value the kernel makes: `hi = f16(q)`, `lo =
    /// f16(q − hi)`, each widened back.
    fn split(q: f32) -> (f32, f32) {
        let hi = half_to_f32(f32_to_f16_bits(q));
        let lo = half_to_f32(f32_to_f16_bits(q - hi));
        (hi, lo)
    }

    /// Our rule over one token's pools, and its bands per pool.
    struct Rule {
        /// `Σ_h w_h · relu(Σ_d (hi + lo)·k)`: the kernel's split query, its
        /// scaled weights, the exact f16 products summed in f64.
        score: Vec<f64>,
        /// The kernel's distance to `score`: per head [`TC_UNITS`] of the
        /// products' magnitude and one rounding of `acc_hi + acc_lo`,
        /// carried by `|w_h|`; then the head sum's roundings.
        kernel: Vec<f64>,
        /// The split's distance to the f32 query.
        rep: Vec<f64>,
        /// ik's distance to the exact score of the f32 query: a 128-long f32
        /// dot per head, its head sum, one final rounding.
        ik: Vec<f64>,
    }

    /// The rule of token `t` of `tokens` (the gemvs' layout: value `i` of
    /// the query at `i·tokens + t`) over the first `n` pools of `keys`, the
    /// weights scaled by `scale` in f32 as the kernel does; `mutant` 1 scales
    /// by `1/√128` instead, 2 drops `relu`, 3 weights head `h` by head `h +
    /// 1`'s weight.
    #[allow(
        clippy::too_many_arguments,
        reason = "the rule's inputs as the launch takes them, and the mutant it runs"
    )]
    fn rule(
        q: &[f32],
        w: &[f32],
        tokens: usize,
        t: usize,
        keys: &[u16],
        n: usize,
        scale: f32,
        mutant: u8,
    ) -> Rule {
        let sp: Vec<[f64; 3]> = (0..HEADS * DIM)
            .map(|i| {
                let v = q[i * tokens + t];
                let (hi, lo) = split(v);
                [f64::from(hi), f64::from(lo), f64::from(v)]
            })
            .collect();
        let sc = if mutant == 1 {
            1.0 / 128f32.sqrt()
        } else {
            scale
        };
        let w64: Vec<f64> = (0..HEADS)
            .map(|h| {
                let hh = if mutant == 3 { (h + 1) % HEADS } else { h };
                f64::from(w[hh * tokens + t] * sc)
            })
            .collect();
        let mut out = Rule {
            score: Vec::with_capacity(n),
            kernel: Vec::with_capacity(n),
            rep: Vec::with_capacity(n),
            ik: Vec::with_capacity(n),
        };
        for j in 0..n {
            let k: Vec<f64> = keys[j * DIM..(j + 1) * DIM]
                .iter()
                .map(|&b| f64::from(half_to_f32(b)))
                .collect();
            let (mut s, mut kb, mut rep, mut ikb, mut smag) = (0.0f64, 0.0, 0.0, 0.0, 0.0);
            for (h, &wh) in w64.iter().enumerate() {
                let (mut dot, mut p, mut pq, mut r) = (0.0f64, 0.0, 0.0, 0.0);
                for (&[hi, lo, qv], &kv) in sp[h * DIM..(h + 1) * DIM].iter().zip(&k) {
                    dot += (hi + lo) * kv;
                    p += (hi * kv).abs() + (lo * kv).abs();
                    pq += (qv * kv).abs();
                    r += (qv - hi - lo).abs() * kv.abs();
                }
                let act = if mutant == 2 { dot } else { dot.max(0.0) };
                s += wh * act;
                smag += wh.abs() * act.abs();
                kb += wh.abs() * (TC_UNITS * U * p + 2.0 * U * dot.abs());
                rep += wh.abs() * r;
                ikb += wh.abs() * gamma(DIM) * pq;
            }
            let hs = gamma(HEAD_SUM_ROUNDINGS);
            out.score.push(s);
            out.kernel.push((1.0 + hs) * kb + hs * smag);
            out.rep.push(rep);
            out.ik.push(ikb + gamma(HEADS) * smag + U * s.abs());
        }
        out
    }

    /// The synthetic score case's counts: one row that sees 512 pools and
    /// is not scored, then three that are.
    const SCORE_COUNTS: [u32; 4] = [2051, 2052, 9001, 16_384];

    /// Clause 2; returns its scores for clause 3.
    fn score_case(cx: &Ctx<'_>) -> Result<(bool, Vec<f32>), GateError> {
        let tokens = SCORE_COUNTS.len();
        let pools = pools_for(CTX);
        let mut r = Lcg(0x7363_6f72);
        let plane = r.f16(pools * DIM, -1.0, 1.0);
        let q = r.fill(HEADS * DIM * tokens, -1.0, 1.0);
        let w = r.fill(HEADS * tokens, -1.0, 1.0);
        let scale = weights_scale(HEADS, DIM);
        let sink = cx.gpu.unlabelled_sink();
        let sc = cx.score(sink, &q, &w, &SCORE_COUNTS, &plane, CTX, scale)?;
        let again = cx.score(sink, &q, &w, &SCORE_COUNTS, &plane, CTX, scale)?;
        let rerun = bits_equal(&sc, &again);
        let mut alone = true;
        for t in 0..tokens {
            let qt: Vec<f32> = (0..HEADS * DIM).map(|i| q[i * tokens + t]).collect();
            let wt: Vec<f32> = (0..HEADS).map(|h| w[h * tokens + t]).collect();
            let one = cx.score(sink, &qt, &wt, &SCORE_COUNTS[t..=t], &plane, CTX, scale)?;
            alone &= bits_equal(&one, &sc[t * pools..(t + 1) * pools]);
        }
        let word = cx.gpu.take_fault()?;
        let (mut worst, mut poison_ok) = (0.0f64, true);
        let mut killed = [false; 3];
        for (t, &c) in SCORE_COUNTS.iter().enumerate() {
            let row = &sc[t * pools..(t + 1) * pools];
            let scored = Qsa::new((KEPT * POOL) as u32, POOL as u32)
                .map_err(|e| e.to_string())?
                .scored(c as usize, CTX);
            let n = if scored { c as usize / POOL } else { 0 };
            poison_ok &= row[n..].iter().all(|v| v.is_nan());
            if n == 0 {
                continue;
            }
            let ru = rule(&q, &w, tokens, t, &plane, n, scale, 0);
            for j in 0..n {
                let d = (f64::from(row[j]) - ru.score[j]).abs();
                worst = worst.max(if d.is_nan() {
                    f64::INFINITY
                } else {
                    d / ru.kernel[j]
                });
            }
            for (mi, k) in killed.iter_mut().enumerate() {
                let mu = rule(&q, &w, tokens, t, &plane, n, scale, mi as u8 + 1);
                *k |= (0..n).any(|j| (mu.score[j] - f64::from(row[j])).abs() > ru.kernel[j]);
            }
        }
        let pass =
            worst <= 1.0 && poison_ok && rerun && alone && word.is_none() && killed == [true; 3];
        println!(
            "score counts {SCORE_COUNTS:?} over {pools} pools: worst |kernel − rule| / band \
             {worst:.3} (<= 1); slots past each row's pools NaN {poison_ok}; rerun bit for bit \
             {rerun}; each token alone its column bit for bit {alone}; fault={}; mutants past the \
             band (scale 1/√128, no relu, head h+1's weight) {killed:?} {}",
            fault_text(word),
            verdict(pass)
        );
        Ok((pass, sc))
    }

    // ------------------------------------------------------------ 3. top-k

    /// Row `t`'s list by the rule: `select_by` of its first `c / 4` scores,
    /// the higher pool of a tie; empty for a refused count.
    fn rule_list(q: Qsa, c: usize, ctx: usize, scores: &[f32], tie: Tie) -> Vec<u32> {
        if c == 0 || c > ctx {
            return Vec::new();
        }
        let n = if q.selects(c) { c / POOL } else { 0 };
        select_by(q, c, &scores[..n], tie)
    }

    /// Clause 3 over clause 2's scores and a planted row of ties.
    fn topk_case(cx: &Ctx<'_>, sc: &[f32]) -> Result<bool, GateError> {
        let q = Qsa::new((KEPT * POOL) as u32, POOL as u32).map_err(|e| e.to_string())?;
        let pools = pools_for(CTX);
        // The planted row at count 8,192 (2,048 pools): 293 distinct
        // positive scores, 136 more equal at 0.25, and relu zeros of both
        // signs for the other 1,619, so the 512th key is a zero and the cut
        // takes 83 of the 1,619 tied pools.
        let tie_count = 8192usize;
        let mut ties = vec![0.0f32; pools];
        for (j, v) in ties.iter_mut().enumerate().take(tie_count / POOL) {
            *v = if j % 7 == 0 {
                (j as f32).mul_add(1e-3, 0.5)
            } else if j % 13 == 5 {
                0.25
            } else if j % 2 == 1 {
                -0.0
            } else {
                0.0
            };
        }
        let mut counts: Vec<u32> = SCORE_COUNTS.to_vec();
        let mut scores = sc.to_vec();
        counts.push(tie_count as u32);
        scores.extend(&ties);
        for c in [1u32, 2051] {
            counts.push(c);
            scores.extend(std::iter::repeat_n(f32::NAN, pools));
        }
        let refused = [0u32, (CTX + 1) as u32];
        let (list, lens) = cx.topk(cx.gpu.unlabelled_sink(), &counts, &scores, CTX)?;
        let word = cx.gpu.take_fault()?;
        let mut ok = word.is_none();
        let mut mism = Vec::new();
        let mut differs = [false; 4];
        for (t, &c) in counts.iter().enumerate() {
            let c = c as usize;
            let row_sc = &scores[t * pools..(t + 1) * pools];
            let want = rule_list(q, c, CTX, row_sc, Tie::Higher);
            let len = lens[t] as usize;
            let got = &list[t * WIDTH..t * WIDTH + len.min(WIDTH)];
            let rest_ok = list[t * WIDTH + len.min(WIDTH)..(t + 1) * WIDTH]
                .iter()
                .all(|&v| v == LIST_POISON);
            if got != want.as_slice() || len != want.len() || !rest_ok {
                mism.push(c);
                ok = false;
            }
            if q.selects(c) {
                let n = c / POOL;
                let lower = select_by(q, c, &row_sc[..n], Tie::Lower);
                differs[0] |= lower != got;
                let q513 = Qsa::new((513 * POOL) as u32, POOL as u32).map_err(|e| e.to_string())?;
                differs[1] |= rule_list(q513, c, CTX, row_sc, Tie::Higher) != got;
                let no_tail: Vec<u32> = want
                    .iter()
                    .copied()
                    .filter(|&p| (p as usize) < n * POOL)
                    .collect();
                differs[2] |= !c.is_multiple_of(POOL) && no_tail != got;
            } else {
                let from1: Vec<u32> = (1..=c as u32).collect();
                differs[3] |= from1 != got;
            }
        }
        // The refused counts, each in its own layer.
        let mut refusal = Vec::new();
        for (i, &c) in refused.iter().enumerate() {
            let layer = 50 + i;
            let sink = cx.gpu.layer_sink(layer)?;
            let (_, lens) = cx.topk(sink, &[c], &vec![0.0; pools], CTX)?;
            let w = cx.gpu.take_fault()?;
            let want = Fault::at(u32::try_from(layer)?, FaultSite::PoolSelect);
            let r = lens[0] == 0 && w == Some(want);
            ok &= r;
            refusal.push(format!(
                "count {c}: length {} fault {} {r}",
                lens[0],
                fault_text(w)
            ));
        }
        let pass = ok && differs == [true; 4];
        println!(
            "topk counts {counts:?}: every list, its length and the slots past it the rule's \
             (the higher pool of a tie) — mismatched at {mism:?}; fault={}; refused [{}]; \
             mutants differ (lower tie, 513 kept, no tail, identity from 1) {differs:?} {}",
            fault_text(word),
            refusal.join("; "),
            verdict(pass)
        );
        Ok(pass)
    }

    // ------------------------------------------------------- 4. identity

    /// Clause 4.
    fn identity_case(cx: &Ctx<'_>) -> Result<bool, GateError> {
        let counts: [u32; 5] = [1, 64, 65, 1000, 2051];
        let rows = 4096usize;
        let mut r = Lcg(0x6964_656e);
        let s3 = 3.0f32.sqrt();
        let cache = r.f16(rows * LATENT, -s3, s3);
        let q = r.fill(counts.len() * ATT_HEADS * LATENT, -2.4, 2.4);
        let (list, lens) = cx.topk(
            cx.gpu.unlabelled_sink(),
            &counts,
            &vec![f32::NAN; counts.len() * pools_for(CTX)],
            CTX,
        )?;
        let lists_ok = counts.iter().enumerate().all(|(t, &c)| {
            lens[t] == c
                && list[t * WIDTH..t * WIDTH + c as usize]
                    .iter()
                    .copied()
                    .eq(0..c)
        });
        let dense = cx.attend(&q, &cache, &counts, None)?;
        let sel = cx.attend(&q, &cache, &counts, Some(&list))?;
        let shifted: Vec<u32> = list
            .iter()
            .map(|&v| if v == LIST_POISON { v } else { v + 1 })
            .collect();
        let moved = cx.attend(&q, &cache, &counts, Some(&shifted))?;
        let word = cx.gpu.take_fault()?;
        let same = bits_equal(&dense, &sel);
        let mutant = !bits_equal(&dense, &moved);
        let pass = lists_ok && same && mutant && word.is_none();
        println!(
            "identity counts {counts:?}: the lists are 0..c {lists_ok}; the attention over them \
             (33 segments) = the prefix attention over {rows} rows (64 segments) bit for bit \
             {same}; the list shifted by one differs {mutant}; fault={} {}",
            fault_text(word),
            verdict(pass)
        );
        Ok(pass)
    }

    // ---------------------------------------------------------- 5. faults

    /// Clause 5: each plant in its own layer, then a clean launch.
    fn fault_cases(cx: &Ctx<'_>) -> Result<Vec<bool>, GateError> {
        let mut out = Vec::new();
        let rows = 64usize;
        let mut r = Lcg(0x6661_756c);
        let mut cache = r.f16(rows * INDEX_ROW, -2.0, 2.0);
        let ape = r.fill(POOL * INDEX_HEAD, -1.0, 1.0);
        let plane = vec![NAN_F16; pools_for(rows) * INDEX_HEAD];
        let counts: Vec<u32> = (1..=rows as u32).collect();
        let pools = pools_for(CTX);
        let splane = r.f16(pools * DIM, -1.0, 1.0);
        let q = r.fill(HEADS * DIM, -1.0, 1.0);
        let w = r.fill(HEADS, -1.0, 1.0);
        let scale = weights_scale(HEADS, DIM);
        let clean_cache = cache.clone();
        cache[5 * INDEX_ROW + 7] = NAN_F16;
        let mut nan_q = q.clone();
        nan_q[3] = f32::NAN;
        let over = [rows as u32 + 1];
        type Run<'a> = Box<dyn Fn(&Ctx<'_>, FaultSink, bool) -> Result<(), GateError> + 'a>;
        let plants: [(usize, &str, FaultSite, Run<'_>); 3] = [
            (
                60,
                "a NaN pool member",
                FaultSite::PoolSelect,
                Box::new(|cx, sink, planted| {
                    let c = if planted { &cache } else { &clean_cache };
                    cx.pool(sink, c, &ape, &counts, &plane).map(|_| ())
                }),
            ),
            (
                61,
                "a NaN indexer query",
                FaultSite::PoolSelect,
                Box::new(|cx, sink, planted| {
                    let qq = if planted { &nan_q } else { &q };
                    cx.score(sink, qq, &w, &[9001], &splane, CTX, scale)
                        .map(|_| ())
                }),
            ),
            (
                62,
                "a pool count past the cache",
                FaultSite::CachePos,
                Box::new(|cx, sink, planted| {
                    let c: &[u32] = if planted { &over } else { &counts };
                    cx.pool(sink, &clean_cache, &ape, c, &plane).map(|_| ())
                }),
            ),
        ];
        for (layer, what, site, run) in &plants {
            let sink = cx.gpu.layer_sink(*layer)?;
            let before = cx.gpu.fault()?;
            run(cx, sink, true)?;
            let word = cx.gpu.take_fault()?;
            run(cx, sink, false)?;
            let after = cx.gpu.take_fault()?;
            let want = Fault::at(u32::try_from(*layer)?, *site);
            let pass = before.is_none() && word == Some(want) && after.is_none();
            println!(
                "fault {what}: word \"{}\" (want \"{want}\"), clean before {} and after {} {}",
                fault_text(word),
                before.is_none(),
                after.is_none(),
                verdict(pass)
            );
            out.push(pass);
        }
        Ok(out)
    }

    // ------------------------------------------------- 7–10. ik's sets

    /// A tensor row of `man`, in its logical order.
    fn tap(man: &RefManifest, name: &str) -> Result<Vec<f32>, GateError> {
        Ok(ref_tensor_logical_in(&man.dir, man.tensor(name, 0)?)?)
    }

    /// The cells ik attends at layer `l` (`dsa_top_k-L`), each below
    /// `bound`: the logical twin's ids, or the plain integer twin's when the
    /// row is contiguous (its plain file is then its logical order).
    fn ik_cells(man: &RefManifest, l: usize, bound: u32) -> Result<Vec<u32>, GateError> {
        let row = man.tensor(&format!("dsa_top_k-{l}"), 0)?;
        if row.logical == Some(1) {
            return Ok(topk_ids_logical_within(man, row, bound)?
                .into_iter()
                .map(|x| x as u32)
                .collect());
        }
        if !plain_is_logical(row) {
            return Err(
                format!("{}: neither a logical twin nor a contiguous row", row.name).into(),
            );
        }
        ref_ints(man, &row.name, row.occurrence, row.kind, Layout::Flat)?
            .into_iter()
            .map(|x| match u32::try_from(x) {
                Ok(c) if c < bound => Ok(c),
                _ => Err(format!("{} holds cell {x}, outside 0..{bound}", row.name).into()),
            })
            .collect()
    }

    /// ik's pool scores at layer `l`: `v` complete pools, finite, and the
    /// slots past them `-inf` (the graph's mask on the pool still filling).
    fn ik_scores(man: &RefManifest, l: usize, v: usize) -> Result<Vec<f32>, GateError> {
        let row = man.tensor(&format!("dsa_indexer_score-{l}"), 0)?;
        Ok(ref_tensor_logical_masked_in(&man.dir, row, v)?)
    }

    /// The f32 values `v` as f16 bits, refused by name unless each is an f16
    /// value exactly: they are ik's f16 cache read back.
    fn exact_f16(what: &str, v: &[f32]) -> Result<Vec<u16>, GateError> {
        v.iter()
            .enumerate()
            .map(|(i, &x)| {
                let h = f32_to_f16_bits(x);
                if half_to_f32(h).to_bits() == x.to_bits() {
                    Ok(h)
                } else {
                    Err(
                        format!("{what}[{i}] = {x:e} is not an f16 value: not ik's f16 cache")
                            .into(),
                    )
                }
            })
            .collect()
    }

    /// Half an f16 ulp at `x`.
    fn half_ulp_f16(x: f64) -> f64 {
        let a = x.abs().max(2f64.powi(-14));
        2f64.powi(a.log2().floor() as i32 - 11)
    }

    /// Clauses 7–10 on set `set`, every latent layer.
    fn ik_set(cx: &Ctx<'_>, split: &Split, set: &str) -> Result<Vec<bool>, GateError> {
        let man = RefManifest::open(&data_dir().join(set), &IK_DSA)?;
        let flags = man.header.flags.as_deref().unwrap_or("");
        if !flags
            .split_whitespace()
            .any(|f| f == "--dsa" || f == "-dsa")
            || man
                .find(
                    bloomery_gpu_gates::RowKind::Tensor,
                    "dsa_indexer_score-3",
                    0,
                )
                .is_none()
        {
            return Err(format!(
                "{set}: not a --dsa dump (flags \"{flags}\"; a dsa_indexer_score-3 row: {}) — \
                 dump it with `just dump-ref-glm5next {}`",
                man.find(
                    bloomery_gpu_gates::RowKind::Tensor,
                    "dsa_indexer_score-3",
                    0
                )
                .is_some(),
                set.trim_start_matches("ref_glm5next_")
            )
            .into());
        }
        let (pos, _, _) = man.step()?;
        let c = pos as usize + 1;
        if c % POOL != POOL - 1 {
            return Err(format!(
                "{set}: step at position {pos}, whose tail is not three cells: ik's zero-filled \
                 tail slots would enter its list"
            )
            .into());
        }
        let v = c / POOL;
        let mut out = Vec::new();
        let mut separated = 0usize;
        for l in LAYERS {
            let ape = split_f32(split, &names::indexer_compressor_ape(l), POOL * INDEX_HEAD)?;
            let pk = tap(&man, &format!("dsa_pool_k-{l}"))?;
            let pg = tap(&man, &format!("dsa_pool_g-{l}"))?;
            let pooled = tap(&man, &format!("dsa_indexer_k_pooled-{l}"))?;
            let n_pool = pooled.len() / DIM;
            if pk.len() != n_pool * POOL * DIM || pg.len() != pk.len() || n_pool < v {
                return Err(format!(
                    "{set} layer {l}: members {} and {} values for {n_pool} pools, {v} seen",
                    pk.len(),
                    pg.len()
                )
                .into());
            }
            // Row 4j + i of our cache is member i of pool j: its key, its gate.
            let mut rows_f = Vec::with_capacity(POOL * v * INDEX_ROW);
            for j in 0..v {
                for i in 0..POOL {
                    let at = (j * POOL + i) * DIM;
                    rows_f.extend_from_slice(&pk[at..at + DIM]);
                    rows_f.extend_from_slice(&pg[at..at + DIM]);
                }
            }
            let cache = exact_f16(&format!("{set} dsa_pool_k/g-{l}"), &rows_f)?;
            out.push(ik_pool(cx, set, l, (&cache, &ape), &pooled, v)?);
            let (pass, ours) = ik_score(cx, &man, set, l, &pooled, (c, n_pool, v))?;
            out.push(pass);
            out.push(ik_select(cx, &man, set, l, &ours, (c, n_pool, v))?);
            let (pass, apart) = ik_attn(cx, &man, set, l, c)?;
            out.push(pass);
            separated += usize::from(apart);
        }
        // How far the dropped pools move a layer's output is the model's,
        // not ours; the clause needs the mutant red on one layer at least.
        let pass = separated >= 1;
        println!(
            "ik attn {set} mutant every position: past twice the band on {separated} of {} \
             layers (>= 1) {}",
            LAYERS.len(),
            verdict(pass)
        );
        out.push(pass);
        Ok(out)
    }

    /// Clause 7 at layer `l`: our pool pass over ik's members against ik's
    /// pooled keys.
    fn ik_pool(
        cx: &Ctx<'_>,
        set: &str,
        l: usize,
        (cache, ape): (&[u16], &[f32]),
        pooled: &[f32],
        v: usize,
    ) -> Result<bool, GateError> {
        let counts: Vec<u32> = (1..=v as u32).map(|j| j * POOL as u32).collect();
        let plane = vec![NAN_F16; v * INDEX_HEAD];
        let card = cx.pool(cx.gpu.unlabelled_sink(), cache, ape, &counts, &plane)?;
        let word = cx.gpu.take_fault()?;
        let bound = |j: usize, d: usize, x: f64| -> f64 {
            let kmax = (0..POOL)
                .map(|i| f64::from(half_to_f32(cache[(j * POOL + i) * INDEX_ROW + d])).abs())
                .fold(0.0, f64::max);
            let slack = POOL_UNITS * U * kmax;
            half_ulp_f16(x.abs() + slack) + slack
        };
        let mut worst = 0.0f64;
        for j in 0..v {
            for d in 0..INDEX_HEAD {
                let ik = f64::from(pooled[j * DIM + d]);
                let ours = f64::from(half_to_f32(card[j * INDEX_HEAD + d]));
                let r = (ours - ik).abs() / bound(j, d, ik);
                worst = worst.max(if r.is_nan() { f64::INFINITY } else { r });
            }
        }
        let mut killed = true;
        let mut moved = Vec::new();
        for (name, mu) in [
            ("no_ape", PoolMut::NoApe),
            ("ape_transposed", PoolMut::ApeTransposed),
            ("plain_mean", PoolMut::PlainMean),
        ] {
            let past = (0..v).any(|j| {
                let m = pool_mutant(cache, ape, j, mu);
                (0..INDEX_HEAD).any(|d| {
                    let ik = f64::from(pooled[j * DIM + d]);
                    (f64::from(m[d]) - ik).abs() > bound(j, d, ik)
                })
            });
            killed &= past;
            moved.push(format!("{name}={past}"));
        }
        let pass = worst <= 1.0 && word.is_none() && killed;
        println!(
            "ik pool {set} layer {l}: {v} pools, worst |ours − ik| / (half an f16 ulp + {POOL_UNITS}u \
             of the largest member) {worst:.3} (<= 1); fault={}; mutants past it [{}] {}",
            fault_text(word),
            moved.join(" "),
            verdict(pass)
        );
        Ok(pass)
    }

    /// Clause 8's scores of one layer's `V` pools and its band per pool,
    /// which clause 9 reads.
    struct Scored {
        scores: Vec<f32>,
        bands: Vec<f64>,
    }

    /// Clause 8 at layer `l`: our score pass on ik's query, weights and
    /// pooled keys (rounded to f16) against ik's scores.
    fn ik_score(
        cx: &Ctx<'_>,
        man: &RefManifest,
        set: &str,
        l: usize,
        pooled: &[f32],
        (c, n_pool, v): (usize, usize, usize),
    ) -> Result<(bool, Scored), GateError> {
        let q = tap(man, &format!("dsa_indexer_q-{l}"))?;
        let w = tap(man, &format!("dsa_indexer_weights-{l}"))?;
        let ik = ik_scores(man, l, v)?;
        if q.len() != HEADS * DIM || w.len() != HEADS || ik.len() < v {
            return Err(format!(
                "{set} layer {l}: query {} weights {} scores {}, want {} {} >= {v}",
                q.len(),
                w.len(),
                ik.len(),
                HEADS * DIM,
                HEADS
            )
            .into());
        }
        let plane: Vec<u16> = pooled.iter().map(|&x| f32_to_f16_bits(x)).collect();
        let ctx = n_pool * POOL;
        let sc = cx.score(
            cx.gpu.unlabelled_sink(),
            &q,
            &w,
            &[c as u32],
            &plane,
            ctx,
            1.0,
        )?;
        let word = cx.gpu.take_fault()?;
        let ru = rule(&q, &w, 1, 0, &plane, v, 1.0, 0);
        let mut bands = Vec::with_capacity(v);
        let mut worst = 0.0f64;
        for j in 0..v {
            // The pools' f16 rounding, carried by |q| and |w|.
            let mut pr = 0.0f64;
            for h in 0..HEADS {
                let dq: f64 = (0..DIM)
                    .map(|d| {
                        let p = f64::from(pooled[j * DIM + d]);
                        let ph = f64::from(half_to_f32(plane[j * DIM + d]));
                        f64::from(q[h * DIM + d]).abs() * (ph - p).abs()
                    })
                    .sum();
                pr += f64::from(w[h]).abs() * dq;
            }
            let band = ru.kernel[j] + ru.rep[j] + pr + ru.ik[j];
            bands.push(band);
            let d = (f64::from(sc[j]) - f64::from(ik[j])).abs();
            worst = worst.max(if d.is_nan() { f64::INFINITY } else { d / band });
        }
        let pass = worst <= 1.0 && word.is_none();
        println!(
            "ik score {set} layer {l}: {v} pools, worst |ours − ik| / band {worst:.3} (<= 1; the \
             band: tensor cores, the split, the pools' f16 rounding, ik's f32 dot) fault={} {}",
            fault_text(word),
            verdict(pass)
        );
        Ok((
            pass,
            Scored {
                scores: sc[..v].to_vec(),
                bands,
            },
        ))
    }

    /// Clause 9 at layer `l`.
    fn ik_select(
        cx: &Ctx<'_>,
        man: &RefManifest,
        set: &str,
        l: usize,
        ours: &Scored,
        (c, n_pool, v): (usize, usize, usize),
    ) -> Result<bool, GateError> {
        let ik = ik_scores(man, l, v)?;
        let mut cells = ik_cells(man, l, u32::try_from(n_pool * POOL)?)?;
        cells.sort_unstable();
        let ctx = n_pool * POOL;
        let mut row = vec![f32::NAN; n_pool];
        row[..v].copy_from_slice(&ik[..v]);
        let (list, lens) = cx.topk(cx.gpu.unlabelled_sink(), &[c as u32], &row, ctx)?;
        let got = &list[..lens[0] as usize];
        let exact = got == cells.as_slice();
        // Our own scores' cut against ik's, pool by pool.
        let mut ours_row = vec![f32::NAN; n_pool];
        ours_row[..v].copy_from_slice(&ours.scores);
        let (olist, olens) = cx.topk(cx.gpu.unlabelled_sink(), &[c as u32], &ours_row, ctx)?;
        let word = cx.gpu.take_fault()?;
        let pools_of = |l: &[u32]| -> Vec<usize> {
            let mut p: Vec<usize> = l
                .iter()
                .map(|&x| x as usize / POOL)
                .filter(|&j| j < v)
                .collect();
            p.dedup();
            p
        };
        let (ik_pools, our_pools) = (pools_of(&cells), pools_of(&olist[..olens[0] as usize]));
        let cut = ik_pools
            .iter()
            .map(|&j| f64::from(ik[j]))
            .fold(f64::INFINITY, f64::min);
        let sym: Vec<usize> = ik_pools
            .iter()
            .filter(|j| !our_pools.contains(j))
            .chain(our_pools.iter().filter(|j| !ik_pools.contains(j)))
            .copied()
            .collect();
        let near = sym
            .iter()
            .all(|&j| (f64::from(ik[j]) - cut).abs() <= 2.0 * ours.bands[j]);
        let pass = exact && lens[0] as usize == WIDTH && near && word.is_none();
        println!(
            "ik select {set} layer {l}: on ik's scores our list = ik's {} cells exactly {exact} \
             (length {}); on our scores {} pools differ from ik's, all within twice the score \
             band of ik's cut {cut:.4e}: {near}; fault={} {}",
            cells.len(),
            lens[0],
            sym.len(),
            fault_text(word),
            verdict(pass)
        );
        Ok(pass)
    }

    /// `gate_glm_mla`'s rule over one token's listed keys: the query rounded
    /// to f16, each logit the exact f64 dot rounded to f32 and scaled in f32,
    /// the softmax and value sum in f64, no sink; and the RMS of the unscaled
    /// logits and the mean |value| its band reads.
    fn attn_rule(q: &[f32], keys: &[f32], list: &[u32]) -> (Vec<f32>, f64, f64) {
        let mut out = Vec::with_capacity(ATT_HEADS * LATENT);
        let (mut sq, mut nl) = (0.0f64, 0usize);
        for h in 0..ATT_HEADS {
            let qr: Vec<f64> = q[h * LATENT..(h + 1) * LATENT]
                .iter()
                .map(|&v| f64::from(half_to_f32(f32_to_f16_bits(v))))
                .collect();
            let logits: Vec<f64> = list
                .iter()
                .map(|&r| {
                    let k = &keys[r as usize * LATENT..][..LATENT];
                    let dot: f64 = k.iter().zip(&qr).map(|(&a, &b)| f64::from(a) * b).sum();
                    sq += dot * dot;
                    nl += 1;
                    f64::from((dot as f32) * ATT_SCALE)
                })
                .collect();
            let mx = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let mut denom = 0.0f64;
            let mut num = vec![0.0f64; LATENT];
            for (&r, &s) in list.iter().zip(&logits) {
                let w = (s - mx).exp();
                denom += w;
                for (a, &v) in num.iter_mut().zip(&keys[r as usize * LATENT..][..LATENT]) {
                    *a += w * f64::from(v);
                }
            }
            out.extend(num.iter().map(|&a| (a / denom) as f32));
        }
        let mean_abs_v = list
            .iter()
            .flat_map(|&r| &keys[r as usize * LATENT..][..LATENT])
            .map(|&v| f64::from(v.abs()))
            .sum::<f64>()
            / (list.len() * LATENT).max(1) as f64;
        (out, (sq / nl.max(1) as f64).sqrt(), mean_abs_v)
    }

    /// Clause 10 at layer `l`.
    fn ik_attn(
        cx: &Ctx<'_>,
        man: &RefManifest,
        set: &str,
        l: usize,
        c: usize,
    ) -> Result<(bool, bool), GateError> {
        let q = tap(man, &format!("Qcur-{l}"))?;
        let kqv = tap(man, &format!("kqv_compressed-{l}"))?;
        let cache = widened_f16_rows_in(&man.dir, man.tensor(&format!("kv_cache-{l}"), 0)?)?;
        let rows = cache.len() / LATENT;
        let mut cells = ik_cells(man, l, u32::try_from(rows)?)?;
        cells.sort_unstable();
        cells.dedup();
        if q.len() != ATT_HEADS * LATENT || kqv.len() != ATT_HEADS * LATENT || rows < c {
            return Err(format!(
                "{set} layer {l}: Qcur {} kqv_compressed {} cache rows {rows} at count {c}",
                q.len(),
                kqv.len()
            )
            .into());
        }
        let mut list = vec![LIST_POISON; WIDTH];
        list[..cells.len()].copy_from_slice(&cells);
        let y = cx.attend(&q, &cache, &[cells.len() as u32], Some(&list))?;
        // The mutant: every position up to the step's, as the dense
        // attention would read past the limit.
        let dense = cx.attend(&q, &cache[..c * LATENT], &[c as u32], None)?;
        let word = cx.gpu.take_fault()?;
        let keys: Vec<f32> = cache.iter().map(|&b| half_to_f32(b)).collect();
        let (want, logit_rms, mean_abs_v) = attn_rule(&q, &keys, &cells);
        let y_max = want.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        let segs = attn::source_segments(cells.len());
        let band = (4.0
            * (U * (LATENT / 8) as f64 * f64::from(ATT_SCALE) * logit_rms * mean_abs_v
                + 2f64.powi(-22) * mean_abs_v
                + U * ((attn::SEG_KEYS as f64).sqrt() + (segs as f64).sqrt()) * mean_abs_v)
            / f64::from(y_max)) as f32;
        let ours = max_rel_err(&y, &want)?;
        let theirs = max_rel_err(&kqv, &want)?;
        let between = max_rel_err(&y, &kqv)?;
        let dense_rel = max_rel_err(&dense, &want)?;
        let pass = ours <= band && word.is_none();
        println!(
            "ik attn {set} layer {l}: {} listed keys; ours vs the rule {ours:.2e}, ik's vs the \
             rule {theirs:.2e} (printed), ours vs ik's {between:.2e} (printed); band {band:.2e} \
             (ours <= it); mutant every position vs the rule {dense_rel:.2e} (printed) fault={} {}",
            cells.len(),
            fault_text(word),
            verdict(pass)
        );
        Ok((pass, dense_rel > 2.0 * band))
    }
}
