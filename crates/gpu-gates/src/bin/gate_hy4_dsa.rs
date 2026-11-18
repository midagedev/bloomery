//! GPU gate for Hy4-flash's k-pool selector on the card (`attention.indexer.kpool_select_tail`): the
//! key append (`latent::index_key_rms_append`), the pool (`latent::index_pool_norm_rope`), the score
//! at a launch's head count (`kpool::kpool_score_tail`) and the top-k that always lists the row's own
//! pool (`qsa::qsa_topk_tail`). Every clause runs on synthetic inputs from fixed seeds at the file's
//! shapes (16 indexer heads of 128, pools of 4, 511 pools kept and the forced tail, lists of 2,048,
//! a NORM rope of the last 64 values at base 10⁷), against two references: the host rule of each
//! kernel (its own arithmetic, op for op: bits), and the exact rule in f64 over the same inputs
//! within a band derived from the f32 arithmetic.
//!
//! Clauses:
//! 1. `append`: 48 tokens at scattered positions of a 4,096-row cache, the raw key at column 16 of
//!    a 160-wide row: every bit of the NaN-poisoned cache the host rule's
//!    (`index_key_rms_append_host`: the written rows, the rest untouched), no fault; every value
//!    within its band of the f64 rule (the RMS norm with its gain, then the value and gate
//!    projections, exact) — half an f16 ulp and [`APPEND_UNITS`] units of the dot's magnitude. Each
//!    rule mutant (LayerNorm's centring, no gain, value and gate swapped, each projection
//!    transposed) must sit past the band at some value.
//! 2. `pool`: the pool over every count `1 ..= 2,048` of a random 2,048-row index cache in one
//!    launch, into a NaN plane: every bit the host rule's (`index_pool_norm_rope_host`), no fault;
//!    every value within its band of the f64 rule (the gated softmax of four with the position
//!    bias, the RMS norm with its gain, the NORM turn of the last 64 values at the pool's first
//!    position by the table's cos and sin). Each rule mutant (no position bias, the plain mean, the
//!    turn at the pool's last position, no norm) must sit past the band at some value.
//! 3. `ring`: 2,048 raw keys appended and pooled densely (one launch each, a 2,048-row cache) and
//!    marched through a ring of 64 index rows in launches of 61 positions (each append, then its
//!    counts' pools): the two planes bit for bit, and both the host rules' chain; a launch of 62
//!    counts over that ring is refused by name.
//! 4. `score`: the score pass at 16 heads over a 4,096-row plane for counts 2,048 (sees 511 pools:
//!    not scored), 2,049, 9,000, 9,001 and 16,384, against the f64 rule (the query turned in f64 at
//!    the row's position `c − 1`, the f32-scaled weights, `relu`, exact) within the kernel's band
//!    (`gate_glm_sel`'s tensor-core band over the kernel's turned split query) plus the split's and
//!    the turn's distance to the exact query; every slot from a row's `(c − 1) / 4` pools on, and
//!    every slot of the unscored row, left NaN; each token alone bit for bit its column of the
//!    five-token launch; a rerun bit for bit. Each rule mutant (no turn, the weights' scale `1/√128`,
//!    head `h` weighted by head `h + 1`) must sit past the band at some pool, and GLM's `c / 4`
//!    pools must name a slot the pass leaves NaN.
//! 5. `heads`: at 32 heads under the identity turn (cos 1, sin 0), every pool both passes score of
//!    counts 9,001 and 16,384 bit for bit GLM-5.3's `kpool_score`'s.
//! 6. `topk`: the top-k over clause 4's scores, two planted rows of ties at the cut (relu zeros of
//!    both signs and a duplicated positive score; tails of one and of four) and rows that do not
//!    select (counts 1, 4, 5, 2,048), against the rule (`runtime::qsa::select_pools_by` of the row's
//!    `(c − 1) / 4` scores, the lower pool of a tie, then the tail `4·n .. c`): every list and its
//!    length exactly, the slots past it untouched; a refused count (0, past the cache) gives length
//!    0 and [`FaultSite::PoolSelect`]. As a set, each scored row's kept pools against the f64 rule's
//!    511 best: every pool the two differ by within the two pools' bands of the cut. Mutants (the
//!    higher pool of a tie, 512 kept, GLM's `c / 4` pools with its tail) must each differ from the
//!    card's lists.
//! 7. `fault`: a NaN raw key and a position at the context (the append), a NaN pool member and a
//!    count past the context (the pool), a NaN query (the score) — each raises its site in its layer
//!    alone, clean before and after a clean launch.
//! 8. `refuse`: the launchers refuse by name a head count that is not a multiple of eight, one past
//!    [`MAX_HEADS`], lists narrower than the forced tail's, a ring that cannot hold a pool launch,
//!    and a projection that is not 128×128 q8_0.
//! 9. `shape`: the four entries compile with no local depot.
//!
//! Tiers (`BLOOMERY_TIER`): every clause is synthetic (`tier::sc`) and reads no file; it runs on
//! either tier.

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_hy4_dsa", gate::run())
}

#[cfg(feature = "gpu")]
mod gate {
    use bloomery_gpu::kpool::{
        DIM, HEADS, KpoolKernels, MAX_HEADS, ScoreArgs, TailScoreArgs, weights_scale,
    };
    use bloomery_gpu::latent::{
        INDEX_HEAD, INDEX_ROW, IndexKeyRmsArgs, IndexPoolRopeArgs, IndexProj, IndexProjHost,
        LatentKernels, POOL, ROT, Rows, index_key_rms_append_host, index_pool_norm_rope_host,
        pools_for,
    };
    use bloomery_gpu::qsa::{QsaKernels, TopkTailArgs, scored_tail, tail_width};
    use bloomery_gpu::{DeviceTensor, Fault, FaultSink, FaultSite, Gpu, GpuError};
    use bloomery_gpu_gates::tier;
    use bloomery_gpu_gates::{
        GateError, NAN_F16, bits_equal, checks_failed, no_local_depot, verdict,
    };
    use cuda_core::{CudaStream, DeviceBuffer};
    use gguf::quant::{f32_to_f16_bits, half_to_f32};
    use runtime::qsa::{Tie, select_pools_by};

    /// Pools a scored row keeps (`attention.indexer.top_k` 2,048 over pools of 4, less the row's
    /// own pool), and its longest list.
    const KEPT: usize = 511;
    const WIDTH: usize = tail_width(KEPT);
    const _: () = assert!(WIDTH == 2048 && DIM == INDEX_HEAD && ROT == 64);
    /// Hy4-flash's indexer query heads (`attention.indexer.head_count`).
    const IDX_HEADS: usize = 16;
    /// The score and top-k cases' cache height.
    const CTX: usize = 16_384;
    /// The rope base of the indexer's turn (`rope.freq_base`).
    const ROPE_BASE: f64 = 1e7;
    /// The two RMS norms' epsilon.
    const EPS: f32 = 1e-5;
    /// f32's unit roundoff, 2⁻²⁴.
    const U: f64 = f32::EPSILON as f64 / 2.0;
    /// The tensor-core dot's bound in units of `U` times its products' magnitude [derived,
    /// `gate_glm_sel`'s `TC_UNITS`]: one k-step errs by at most 36u of its addends' magnitude,
    /// eight k-steps a head.
    const TC_UNITS: f64 = 288.0;
    /// Roundings one head's weighted term meets in the score's head sum at 16 heads: a product
    /// and one fused multiply-add along its lane group, then three butterfly levels.
    const HEAD_SUM_ROUNDINGS: usize = 5;
    /// The append's f32 distance to the f64 rule before its f16 rounding, in units of `U` times
    /// the row's `Σ_d |W_rd · k̂_d|` [derived]: the normed key carries the scale's error (the mean's
    /// cast, the epsilon's add, the square root and the reciprocal: 4u) and its two products
    /// (2u); the row's dot is four fused multiply-adds and five butterfly adds, γ(9); 15u, held
    /// at 32 with the margin the other gates' bounds take.
    const APPEND_UNITS: f64 = 32.0;
    /// The pool and score bands' margin over their derived f32 distance.
    const MARGIN: f64 = 2.0;
    /// What the lists are poisoned with.
    const LIST_POISON: u32 = u32::MAX;

    /// `γ(n) = n·u / (1 − n·u)`.
    fn gamma(n: usize) -> f64 {
        let nu = n as f64 * U;
        nu / (1.0 - nu)
    }

    /// Half an f16 ulp at magnitude `x` (subnormals included).
    fn half_ulp_f16(x: f64) -> f64 {
        let x = x.abs();
        if x < 2f64.powi(-14) {
            2f64.powi(-25)
        } else {
            2f64.powi(x.log2().floor() as i32 - 11)
        }
    }

    fn widen(b: u16) -> f64 {
        f64::from(half_to_f32(b))
    }

    /// A 64-bit LCG (Knuth's MMIX constants); the high 24 bits as a unit float.
    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            self.0
        }
        fn unit(&mut self) -> f32 {
            (self.next() >> 40) as f32 / (1u64 << 24) as f32
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

    /// One 128×128 q8_0 projection in the q8f32 planes: 32 code words and 4 f16 scales a row.
    struct Q8 {
        qs: Vec<u32>,
        d: Vec<u16>,
    }

    impl Q8 {
        fn random(r: &mut Lcg) -> Q8 {
            let qs = (0..INDEX_HEAD * INDEX_HEAD / 4)
                .map(|_| {
                    let mut w = 0u32;
                    for b in 0..4 {
                        let code = (r.next() >> 33) % 255;
                        let v = code as i32 - 127;
                        w |= u32::from(v as i8 as u8) << (8 * b);
                    }
                    w
                })
                .collect();
            let d = r.f16(INDEX_HEAD * INDEX_HEAD / 32, 0.002, 0.01);
            Q8 { qs, d }
        }
        /// The weight of row `r`, column `c`, exact.
        fn at(&self, r: usize, c: usize) -> f64 {
            let q = self.qs[r * 32 + c / 4];
            let v = f64::from((q >> (8 * (c % 4))) as u8 as i8);
            widen(self.d[r * 4 + c / 32]) * v
        }
        fn host(&self) -> IndexProjHost<'_> {
            IndexProjHost {
                qs: &self.qs,
                d: &self.d,
            }
        }
    }

    /// The projection's planes on the card.
    struct Q8Dev {
        qs: DeviceTensor<u32>,
        d: DeviceTensor<u16>,
    }

    impl Q8Dev {
        fn upload(s: &CudaStream, w: &Q8) -> Result<Q8Dev, GateError> {
            Ok(Q8Dev {
                qs: DeviceTensor::upload(s, &w.qs, INDEX_HEAD, INDEX_HEAD / 4)?,
                d: DeviceTensor::upload(s, &w.d, INDEX_HEAD, INDEX_HEAD / 32)?,
            })
        }
        fn proj(&self) -> IndexProj<'_> {
            IndexProj {
                qs: &self.qs,
                d: &self.d,
            }
        }
    }

    /// `ctx` rows of the indexer's rope table: row `p` the 32 pairs `[cos θ_i, sin θ_i]` of `θ_i =
    /// p · base^(−2i/64)`, each rounded to f32 once; `identity` writes cos 1 and sin 0 everywhere.
    fn rope_rows(ctx: usize, identity: bool) -> Vec<f32> {
        let mut out = Vec::with_capacity(ctx * ROT);
        for p in 0..ctx {
            for i in 0..ROT / 2 {
                if identity {
                    out.extend([1.0f32, 0.0]);
                } else {
                    let th = p as f64 * ROPE_BASE.powf(-2.0 * i as f64 / ROT as f64);
                    out.extend([th.cos() as f32, th.sin() as f32]);
                }
            }
        }
        out
    }

    /// One NORM pair turned as the kernels' `rope_pair_rn` turns it: `x0·c − x1·s` and `x0·s +
    /// x1·c`, every product and sum rounded on its own (the host never fuses them).
    fn turn(x0: f32, x1: f32, c: f32, s: f32) -> (f32, f32) {
        (x0 * c - x1 * s, x0 * s + x1 * c)
    }

    fn diff_u16(a: &[u16], b: &[u16]) -> usize {
        a.iter().zip(b).filter(|(x, y)| x != y).count() + a.len().abs_diff(b.len())
    }

    fn fault_text(f: Option<Fault>) -> String {
        f.map_or_else(|| "none".to_owned(), |f| f.to_string())
    }

    /// The inputs of one append launch.
    struct AppendIn<'a> {
        x: &'a [f32],
        stride: usize,
        k_off: usize,
        gain: &'a [f32],
        wv: &'a Q8,
        wg: &'a Q8,
        pos: &'a [u32],
        ctx: usize,
    }

    /// The inputs of one pool launch beside the index rows and the counts.
    struct PoolIn<'a> {
        ape: &'a [f32],
        gain: &'a [f32],
        table: &'a [f32],
        ctx: usize,
    }

    struct Ctx<'a> {
        gpu: &'a Gpu,
        lk: LatentKernels,
        kk: KpoolKernels,
        qk: QsaKernels,
    }

    impl Ctx<'_> {
        fn stream(&self) -> &CudaStream {
            self.gpu.stream()
        }

        /// The append of `a` into a copy of `cache` (`[rows][INDEX_ROW]`), read back.
        fn append(
            &self,
            sink: FaultSink,
            a: &AppendIn<'_>,
            cache: &[u16],
        ) -> Result<Vec<u16>, GateError> {
            let s = self.stream();
            let x = DeviceBuffer::from_host(s, a.x)?;
            let gain = DeviceBuffer::from_host(s, a.gain)?;
            let (wv, wg) = (Q8Dev::upload(s, a.wv)?, Q8Dev::upload(s, a.wg)?);
            let pos = DeviceBuffer::from_host(s, a.pos)?;
            let mut c = DeviceTensor::upload(s, cache, cache.len() / INDEX_ROW, INDEX_ROW)?;
            self.lk.enqueue_index_key_rms_append(
                s,
                IndexKeyRmsArgs {
                    rows: Rows {
                        x: &x,
                        stride: a.stride,
                        m: a.pos.len(),
                    },
                    k_off: a.k_off,
                    gain: &gain,
                    value: wv.proj(),
                    gate: wg.proj(),
                    pos: &pos,
                    eps: EPS,
                    ctx: a.ctx,
                    fault: sink,
                    cache: &mut c,
                },
            )?;
            s.synchronize()?;
            Ok(c.buf().to_host_vec(s)?)
        }

        /// The pool of `counts` over `cache` into a copy of `plane`, read back.
        fn pool(
            &self,
            sink: FaultSink,
            cache: &[u16],
            p: &PoolIn<'_>,
            counts: &[u32],
            plane: &[u16],
        ) -> Result<Vec<u16>, GateError> {
            let s = self.stream();
            let cache = DeviceTensor::upload(s, cache, cache.len() / INDEX_ROW, INDEX_ROW)?;
            let ape = DeviceBuffer::from_host(s, p.ape)?;
            let gain = DeviceBuffer::from_host(s, p.gain)?;
            let table = DeviceBuffer::from_host(s, p.table)?;
            let n_keys = DeviceBuffer::from_host(s, counts)?;
            let mut pooled = DeviceTensor::upload(s, plane, plane.len() / INDEX_HEAD, INDEX_HEAD)?;
            self.lk.enqueue_index_pool_norm_rope(
                s,
                IndexPoolRopeArgs {
                    cache: &cache,
                    ape: &ape,
                    gain: &gain,
                    table: &table,
                    n_keys: &n_keys,
                    eps: EPS,
                    ctx: p.ctx,
                    m: counts.len(),
                    fault: sink,
                    pooled: &mut pooled,
                },
            )?;
            s.synchronize()?;
            Ok(pooled.buf().to_host_vec(s)?)
        }

        /// The forced-tail score pass for `counts.len()` tokens at `heads` heads into NaN scores.
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
            table: &[f32],
            plane: &[u16],
            heads: usize,
            scale: f32,
        ) -> Result<Vec<f32>, GateError> {
            let s = self.stream();
            let pools = pools_for(CTX);
            let qb = DeviceBuffer::from_host(s, q)?;
            let wb = DeviceBuffer::from_host(s, w)?;
            let n_keys = DeviceBuffer::from_host(s, counts)?;
            let tb = DeviceBuffer::from_host(s, table)?;
            let pooled = DeviceTensor::upload(s, plane, plane.len() / DIM, DIM)?;
            let mut scores = DeviceBuffer::from_host(s, &vec![f32::NAN; counts.len() * pools])?;
            self.kk.enqueue_score_tail(
                s,
                TailScoreArgs {
                    q: &qb,
                    w: &wb,
                    n_keys: &n_keys,
                    table: &tb,
                    pooled: &pooled,
                    heads,
                    tokens: counts.len(),
                    ctx: CTX,
                    kept: KEPT,
                    scale,
                    fault: sink,
                    scores: &mut scores,
                },
            )?;
            s.synchronize()?;
            Ok(scores.to_host_vec(s)?)
        }

        /// GLM-5.3's score pass (32 heads, `c / 4` pools) at `kept` [`KEPT`] into NaN scores.
        fn score_glm(
            &self,
            q: &[f32],
            w: &[f32],
            counts: &[u32],
            plane: &[u16],
            scale: f32,
        ) -> Result<Vec<f32>, GateError> {
            let s = self.stream();
            let pools = pools_for(CTX);
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
                    ctx: CTX,
                    kept: KEPT,
                    scale,
                    fault: self.gpu.unlabelled_sink(),
                    scores: &mut scores,
                },
            )?;
            s.synchronize()?;
            Ok(scores.to_host_vec(s)?)
        }

        /// The forced-tail top-k for `counts.len()` rows of `scores` into poisoned lists: each
        /// row's list and length.
        fn topk(
            &self,
            sink: FaultSink,
            counts: &[u32],
            scores: &[f32],
        ) -> Result<(Vec<u32>, Vec<u32>), GateError> {
            let s = self.stream();
            let m = counts.len();
            let n_keys = DeviceBuffer::from_host(s, counts)?;
            let sc = DeviceBuffer::from_host(s, scores)?;
            let mut list = DeviceBuffer::from_host(s, &vec![LIST_POISON; m * WIDTH])?;
            let mut n_sel = DeviceBuffer::from_host(s, &vec![LIST_POISON; m])?;
            self.qk.enqueue_topk_tail(
                s,
                TopkTailArgs {
                    n_keys: &n_keys,
                    scores: &sc,
                    ctx: CTX,
                    kept: KEPT,
                    width: WIDTH,
                    m,
                    fault: sink,
                    list: &mut list,
                    n_sel: &mut n_sel,
                },
            )?;
            s.synchronize()?;
            Ok((list.to_host_vec(s)?, n_sel.to_host_vec(s)?))
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
        };
        println!(
            "gate_hy4_dsa: device {} — indexer heads={IDX_HEADS} dim={DIM} pool={POOL} \
             kept={KEPT} width={WIDTH} rope={ROT}@{ROPE_BASE:e}",
            gpu.device_name()?
        );
        gpu.clear_fault()?;
        let mut failed = 0u32;
        let mut clauses = 0u32;
        let mut tally = |pass: bool| {
            clauses += 1;
            failed += u32::from(!pass);
        };
        tier::sc("1 append: the key append against the host rule and the f64 rule")?;
        tally(append_case(&cx)?);
        tier::sc("2 pool: the pool against the host rule and the f64 rule")?;
        tally(pool_case(&cx)?);
        tier::sc("3 ring: a ring of index rows marched in launches equals the dense cache")?;
        tally(ring_case(&cx)?);
        tier::sc("4 score: the forced-tail score at 16 heads against the f64 rule")?;
        let (score_ok, sc) = score_case(&cx)?;
        tally(score_ok);
        tier::sc("5 heads: 32 heads under the identity turn are GLM-5.3's score")?;
        tally(heads_case(&cx)?);
        tier::sc("6 topk: the forced-tail lists against the rule, ties to the lower pool")?;
        tally(topk_case(&cx, &sc)?);
        tier::sc("7 fault: each kernel's fault site, raised in its layer alone")?;
        for pass in fault_cases(&cx)? {
            tally(pass);
        }
        tier::sc("8 refuse: the launchers refuse a shape the entries do not take")?;
        tally(refuse_case(&cx)?);
        tier::sc("9 shape: the four entries compile with no local depot")?;
        tally(no_local_depot(&[
            "index_key_rms_append",
            "index_pool_norm_rope",
            "kpool_score_tail",
            "qsa_topk_tail",
        ])?);
        let pass = failed == 0;
        println!(
            "gate_hy4_dsa: {clauses} clauses, {failed} failed — {}",
            verdict(pass)
        );
        println!("gate_hy4_dsa: {}", tier::tally_line());
        if !pass {
            return Err(checks_failed());
        }
        Ok(())
    }

    // ---------------------------------------------------------- 1. append

    #[derive(Clone, Copy, PartialEq)]
    enum AppendMut {
        Base,
        Centred,
        NoGain,
        Swapped,
        Transposed,
    }

    /// The f64 rule of one token's index row (value then gate) and each value's dot magnitude
    /// `Σ_d |W_rd · k̂_d|`.
    fn append_ref(
        k: &[f32],
        gain: &[f32],
        wv: &Q8,
        wg: &Q8,
        mu: AppendMut,
    ) -> (Vec<f64>, Vec<f64>) {
        let mut kk: Vec<f64> = k.iter().map(|&v| f64::from(v)).collect();
        if mu == AppendMut::Centred {
            let mean = kk.iter().sum::<f64>() / INDEX_HEAD as f64;
            for v in &mut kk {
                *v -= mean;
            }
        }
        let ms = kk.iter().map(|v| v * v).sum::<f64>() / INDEX_HEAD as f64;
        let s = 1.0 / (ms + f64::from(EPS)).sqrt();
        let kn: Vec<f64> = (0..INDEX_HEAD)
            .map(|d| {
                let g = if mu == AppendMut::NoGain {
                    1.0
                } else {
                    f64::from(gain[d])
                };
                kk[d] * s * g
            })
            .collect();
        let halves = if mu == AppendMut::Swapped {
            [wg, wv]
        } else {
            [wv, wg]
        };
        let (mut vals, mut mags) = (Vec::with_capacity(INDEX_ROW), Vec::with_capacity(INDEX_ROW));
        for w in halves {
            for r in 0..INDEX_HEAD {
                let (mut v, mut m) = (0.0f64, 0.0f64);
                for (d, &x) in kn.iter().enumerate() {
                    let wt = if mu == AppendMut::Transposed {
                        w.at(d, r)
                    } else {
                        w.at(r, d)
                    };
                    v += wt * x;
                    m += (wt * x).abs();
                }
                vals.push(v);
                mags.push(m);
            }
        }
        (vals, mags)
    }

    /// Clause 1.
    fn append_case(cx: &Ctx<'_>) -> Result<bool, GateError> {
        let (m, stride, k_off, ctx) = (48usize, 160usize, 16usize, 4096usize);
        let mut r = Lcg(0x6170_706e);
        let x = r.fill(m * stride, -2.0, 2.0);
        let gain = r.fill(INDEX_HEAD, 0.5, 1.5);
        let (wv, wg) = (Q8::random(&mut r), Q8::random(&mut r));
        let pos: Vec<u32> = (0..m).map(|t| (37 * t + 5) as u32).collect();
        let a = AppendIn {
            x: &x,
            stride,
            k_off,
            gain: &gain,
            wv: &wv,
            wg: &wg,
            pos: &pos,
            ctx,
        };
        let init = vec![NAN_F16; ctx * INDEX_ROW];
        let card = cx.append(cx.gpu.unlabelled_sink(), &a, &init)?;
        let word = cx.gpu.take_fault()?;
        let mut want = init;
        let sites = index_key_rms_append_host(
            &x,
            stride,
            k_off,
            &gain,
            (wv.host(), wg.host()),
            &pos,
            EPS,
            ctx,
            &mut want,
        );
        let differ = diff_u16(&card, &want);
        let mut worst = 0.0f64;
        let muts = [
            ("centred", AppendMut::Centred),
            ("no_gain", AppendMut::NoGain),
            ("swapped", AppendMut::Swapped),
            ("transposed", AppendMut::Transposed),
        ];
        let mut killed = [false; 4];
        for (t, &p) in pos.iter().enumerate() {
            let k = &x[t * stride + k_off..][..INDEX_HEAD];
            let row = &card[p as usize * INDEX_ROW..][..INDEX_ROW];
            let (want64, mags) = append_ref(k, &gain, &wv, &wg, AppendMut::Base);
            let band: Vec<f64> = (0..INDEX_ROW)
                .map(|i| {
                    half_ulp_f16(widen(row[i]).abs().max(want64[i].abs()) * (1.0 + 1e-3))
                        + APPEND_UNITS * U * mags[i]
                })
                .collect();
            for i in 0..INDEX_ROW {
                let d = (widen(row[i]) - want64[i]).abs();
                worst = worst.max(if d.is_nan() {
                    f64::INFINITY
                } else {
                    d / band[i]
                });
            }
            for (k_i, &(_, mu)) in muts.iter().enumerate() {
                let (mv, _) = append_ref(k, &gain, &wv, &wg, mu);
                killed[k_i] |= (0..INDEX_ROW).any(|i| (widen(row[i]) - mv[i]).abs() > band[i]);
            }
        }
        let pass =
            differ == 0 && sites == 0 && word.is_none() && worst <= 1.0 && killed == [true; 4];
        println!(
            "append {m} tokens at positions 5, 42, … of {ctx}: cache bits differ from the host rule \
             {differ}/{} host_sites={sites:#x} fault={}; worst |card − f64 rule| / band {worst:.3} \
             (<= 1); mutants past the band ({}) {killed:?} {}",
            card.len(),
            fault_text(word),
            muts.iter().map(|m| m.0).collect::<Vec<_>>().join(", "),
            verdict(pass)
        );
        Ok(pass)
    }

    // ------------------------------------------------------------ 2. pool

    #[derive(Clone, Copy, PartialEq)]
    enum PoolMut {
        Base,
        NoApe,
        PlainMean,
        TurnLast,
        NoNorm,
    }

    /// The f64 rule of pool `j` over `cache` (`[ring][INDEX_ROW]` f16, position `p` in slot `p
    /// mod ring`) and each value's band: half an f16 ulp and [`MARGIN`] times the f32
    /// arithmetic's derived distance (module doc of `latent`'s pool: the exponentials' 2 ulps
    /// and their arguments' roundings, the weights' and the sum's roundings; the norm's scale
    /// error from the pool's; the turn's two products and add).
    fn pool_ref(cache: &[u16], p: &PoolIn<'_>, j: usize, mu: PoolMut) -> (Vec<f64>, Vec<f64>) {
        let ring = cache.len() / INDEX_ROW;
        let row = |i: usize| &cache[((POOL * j + i) % ring) * INDEX_ROW..][..INDEX_ROW];
        let mut pv = vec![0.0f64; INDEX_HEAD];
        let mut pe = vec![0.0f64; INDEX_HEAD];
        for d in 0..INDEX_HEAD {
            let k: Vec<f64> = (0..POOL).map(|i| widen(row(i)[d])).collect();
            let kmax = k.iter().fold(0.0f64, |a, v| a.max(v.abs()));
            if mu == PoolMut::PlainMean {
                pv[d] = k.iter().sum::<f64>() / POOL as f64;
                continue;
            }
            let z: Vec<f64> = (0..POOL)
                .map(|i| {
                    let a = if mu == PoolMut::NoApe {
                        0.0
                    } else {
                        f64::from(p.ape[i * INDEX_HEAD + d])
                    };
                    widen(row(i)[INDEX_HEAD + d]) + a
                })
                .collect();
            let mx = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let e: Vec<f64> = z.iter().map(|v| (v - mx).exp()).collect();
            let s: f64 = e.iter().sum();
            pv[d] = e.iter().zip(&k).map(|(e, k)| e * k).sum::<f64>() / s;
            let eps_max = z
                .iter()
                .map(|v| 4.0 * U + U * (v.abs() + mx.abs() + (v - mx).abs()))
                .fold(0.0f64, f64::max);
            pe[d] = (2.0 * eps_max + 9.0 * U) * kmax;
        }
        let s2: f64 = pv.iter().map(|v| v * v).sum();
        let s = if mu == PoolMut::NoNorm {
            1.0
        } else {
            1.0 / (s2 / INDEX_HEAD as f64 + f64::from(EPS)).sqrt()
        };
        let mean_rel =
            2.0 * pv.iter().zip(&pe).map(|(v, e)| v.abs() * e).sum::<f64>() / s2 + 2.0 * U;
        let ds = 0.5 * (mean_rel + U) + 2.0 * U;
        let mut y: Vec<f64> = (0..INDEX_HEAD)
            .map(|d| pv[d] * s * f64::from(p.gain[d]))
            .collect();
        let mut ye: Vec<f64> = (0..INDEX_HEAD)
            .map(|d| {
                f64::from(p.gain[d]).abs() * s * (pe[d] + pv[d].abs() * ds) + 2.0 * U * y[d].abs()
            })
            .collect();
        let at_pos = if mu == PoolMut::TurnLast {
            POOL * j + POOL - 1
        } else {
            POOL * j
        };
        let tab = &p.table[at_pos * ROT..][..ROT];
        for i in 0..ROT / 2 {
            let at = INDEX_HEAD - ROT + 2 * i;
            let (c, sn) = (f64::from(tab[2 * i]), f64::from(tab[2 * i + 1]));
            let (a, b) = (y[at], y[at + 1]);
            let (ea, eb) = (ye[at], ye[at + 1]);
            y[at] = a * c - b * sn;
            y[at + 1] = a * sn + b * c;
            ye[at] = c.abs() * ea + sn.abs() * eb + 2.0 * U * ((a * c).abs() + (b * sn).abs());
            ye[at + 1] = sn.abs() * ea + c.abs() * eb + 2.0 * U * ((a * sn).abs() + (b * c).abs());
        }
        let band = (0..INDEX_HEAD).map(|d| MARGIN * ye[d]).collect();
        (y, band)
    }

    /// Clause 2.
    fn pool_case(cx: &Ctx<'_>) -> Result<bool, GateError> {
        let rows = 2048usize;
        let mut r = Lcg(0x706f_6f6c);
        let mut cache = Vec::with_capacity(rows * INDEX_ROW);
        for _ in 0..rows {
            cache.extend(r.f16(INDEX_HEAD, -3.0, 3.0));
            cache.extend(r.f16(INDEX_HEAD, -6.0, 6.0));
        }
        let ape = r.fill(POOL * INDEX_HEAD, -1.0, 1.0);
        let gain = r.fill(INDEX_HEAD, 0.5, 1.5);
        let table = rope_rows(rows, false);
        let p = PoolIn {
            ape: &ape,
            gain: &gain,
            table: &table,
            ctx: rows,
        };
        let counts: Vec<u32> = (1..=rows as u32).collect();
        let plane = vec![NAN_F16; pools_for(rows) * INDEX_HEAD];
        let card = cx.pool(cx.gpu.unlabelled_sink(), &cache, &p, &counts, &plane)?;
        let word = cx.gpu.take_fault()?;
        let mut want = plane;
        let sites =
            index_pool_norm_rope_host(&cache, (&ape, &gain, &table), &counts, EPS, rows, &mut want);
        let differ = diff_u16(&card, &want);
        let muts = [
            ("no_ape", PoolMut::NoApe),
            ("plain_mean", PoolMut::PlainMean),
            ("turn_at_last", PoolMut::TurnLast),
            ("no_norm", PoolMut::NoNorm),
        ];
        let mut killed = [false; 4];
        let mut worst = 0.0f64;
        for j in 0..rows / POOL {
            let got = &card[j * INDEX_HEAD..][..INDEX_HEAD];
            let (y, band) = pool_ref(&cache, &p, j, PoolMut::Base);
            for d in 0..INDEX_HEAD {
                let b = band[d] + half_ulp_f16(widen(got[d]).abs().max(y[d].abs()) * (1.0 + 1e-3));
                let e = (widen(got[d]) - y[d]).abs();
                worst = worst.max(if e.is_nan() { f64::INFINITY } else { e / b });
            }
            for (mi, &(_, mu)) in muts.iter().enumerate() {
                let (ym, _) = pool_ref(&cache, &p, j, mu);
                killed[mi] |= (0..INDEX_HEAD).any(|d| {
                    let b =
                        band[d] + half_ulp_f16(widen(got[d]).abs().max(y[d].abs()) * (1.0 + 1e-3));
                    (widen(got[d]) - ym[d]).abs() > b
                });
            }
        }
        let pass =
            differ == 0 && sites == 0 && word.is_none() && worst <= 1.0 && killed == [true; 4];
        println!(
            "pool counts 1..={rows} in one launch: plane bits differ from the host rule {differ}/{} \
             host_sites={sites:#x} fault={}; worst |card − f64 rule| / band {worst:.3} (<= 1); \
             mutants past the band ({}) {killed:?} {}",
            card.len(),
            fault_text(word),
            muts.iter().map(|m| m.0).collect::<Vec<_>>().join(", "),
            verdict(pass)
        );
        Ok(pass)
    }

    // ------------------------------------------------------------ 3. ring

    /// Appends then pools `0..n` in launches of `chunk` positions over an index-row cache of
    /// `ring` rows; returns the plane.
    #[allow(
        clippy::too_many_arguments,
        reason = "the two launches' inputs, held across the march"
    )]
    fn march(
        cx: &Ctx<'_>,
        x: &[f32],
        gain: &[f32],
        (wv, wg): (&Q8, &Q8),
        p: &PoolIn<'_>,
        n: usize,
        ring: usize,
        chunk: usize,
    ) -> Result<Vec<u16>, GateError> {
        let s = cx.stream();
        let kg = DeviceBuffer::from_host(s, gain)?;
        let (wv, wg) = (Q8Dev::upload(s, wv)?, Q8Dev::upload(s, wg)?);
        let ape = DeviceBuffer::from_host(s, p.ape)?;
        let pg = DeviceBuffer::from_host(s, p.gain)?;
        let table = DeviceBuffer::from_host(s, p.table)?;
        let mut cache = DeviceTensor::upload(s, &vec![NAN_F16; ring * INDEX_ROW], ring, INDEX_ROW)?;
        let mut plane = DeviceTensor::upload(
            s,
            &vec![NAN_F16; pools_for(p.ctx) * INDEX_HEAD],
            pools_for(p.ctx),
            INDEX_HEAD,
        )?;
        // Each launch's raw keys, positions and counts, held until the march's last launch has
        // run: the launches read them asynchronously.
        let mut held = Vec::new();
        let mut p0 = 0usize;
        while p0 < n {
            let m = chunk.min(n - p0);
            let pos: Vec<u32> = (p0..p0 + m).map(|v| v as u32).collect();
            let counts: Vec<u32> = (p0 + 1..=p0 + m).map(|v| v as u32).collect();
            let pos = DeviceBuffer::from_host(s, &pos)?;
            let n_keys = DeviceBuffer::from_host(s, &counts)?;
            let xc = DeviceBuffer::from_host(s, &x[p0 * INDEX_HEAD..(p0 + m) * INDEX_HEAD])?;
            cx.lk.enqueue_index_key_rms_append(
                s,
                IndexKeyRmsArgs {
                    rows: Rows {
                        x: &xc,
                        stride: INDEX_HEAD,
                        m,
                    },
                    k_off: 0,
                    gain: &kg,
                    value: wv.proj(),
                    gate: wg.proj(),
                    pos: &pos,
                    eps: EPS,
                    ctx: p.ctx,
                    fault: cx.gpu.unlabelled_sink(),
                    cache: &mut cache,
                },
            )?;
            cx.lk.enqueue_index_pool_norm_rope(
                s,
                IndexPoolRopeArgs {
                    cache: &cache,
                    ape: &ape,
                    gain: &pg,
                    table: &table,
                    n_keys: &n_keys,
                    eps: EPS,
                    ctx: p.ctx,
                    m,
                    fault: cx.gpu.unlabelled_sink(),
                    pooled: &mut plane,
                },
            )?;
            held.push((xc, pos, n_keys));
            p0 += m;
        }
        s.synchronize()?;
        Ok(plane.buf().to_host_vec(s)?)
    }

    /// Clause 3.
    fn ring_case(cx: &Ctx<'_>) -> Result<bool, GateError> {
        let n = 2048usize;
        let mut r = Lcg(0x7269_6e67);
        let x = r.fill(n * INDEX_HEAD, -2.0, 2.0);
        let gain = r.fill(INDEX_HEAD, 0.5, 1.5);
        let (wv, wg) = (Q8::random(&mut r), Q8::random(&mut r));
        let ape = r.fill(POOL * INDEX_HEAD, -1.0, 1.0);
        let pgain = r.fill(INDEX_HEAD, 0.5, 1.5);
        let table = rope_rows(n, false);
        let p = PoolIn {
            ape: &ape,
            gain: &pgain,
            table: &table,
            ctx: n,
        };
        let dense = march(cx, &x, &gain, (&wv, &wg), &p, n, n, n)?;
        let ringed = march(cx, &x, &gain, (&wv, &wg), &p, n, 64, 61)?;
        let word = cx.gpu.take_fault()?;
        let mut hc = vec![NAN_F16; n * INDEX_ROW];
        let pos: Vec<u32> = (0..n as u32).collect();
        let s1 = index_key_rms_append_host(
            &x,
            INDEX_HEAD,
            0,
            &gain,
            (wv.host(), wg.host()),
            &pos,
            EPS,
            n,
            &mut hc,
        );
        let mut hp = vec![NAN_F16; pools_for(n) * INDEX_HEAD];
        let counts: Vec<u32> = (1..=n as u32).collect();
        let s2 = index_pool_norm_rope_host(&hc, (&ape, &pgain, &table), &counts, EPS, n, &mut hp);
        let same = diff_u16(&dense, &ringed);
        let host = diff_u16(&dense, &hp);
        // A launch of 62 counts over a ring of 64 would read a slot its own appends replaced.
        let s = cx.stream();
        let cache = DeviceTensor::upload(s, &vec![NAN_F16; 64 * INDEX_ROW], 64, INDEX_ROW)?;
        let ab = DeviceBuffer::from_host(s, &ape)?;
        let gb = DeviceBuffer::from_host(s, &pgain)?;
        let tb = DeviceBuffer::from_host(s, &table)?;
        let nk = DeviceBuffer::from_host(s, &counts[..62])?;
        let mut plane = DeviceTensor::upload(
            s,
            &vec![NAN_F16; pools_for(n) * INDEX_HEAD],
            pools_for(n),
            INDEX_HEAD,
        )?;
        let refused = cx.lk.enqueue_index_pool_norm_rope(
            s,
            IndexPoolRopeArgs {
                cache: &cache,
                ape: &ab,
                gain: &gb,
                table: &tb,
                n_keys: &nk,
                eps: EPS,
                ctx: n,
                m: 62,
                fault: cx.gpu.unlabelled_sink(),
                pooled: &mut plane,
            },
        );
        let refused_ok = matches!(
            &refused,
            Err(GpuError::Shape { what, .. }) if *what == "enqueue_index_pool_norm_rope"
        );
        let pass = same == 0 && host == 0 && s1 == 0 && s2 == 0 && word.is_none() && refused_ok;
        println!(
            "ring {n} positions: the march over 64 rows in launches of 61 differs from the dense \
             launch at {same}/{} plane bits; the dense plane differs from the host rules' chain at \
             {host} (sites {s1:#x} {s2:#x}); fault={}; 62 counts over 64 rows refused: {} {}",
            dense.len(),
            fault_text(word),
            match &refused {
                Ok(()) => "not refused".to_owned(),
                Err(e) => e.to_string(),
            },
            verdict(pass)
        );
        Ok(pass)
    }

    // ----------------------------------------------------------- 4. score

    /// The split of one query value the kernel makes: `hi = f16(q)`, `lo = f16(q − hi)`, each
    /// widened back.
    fn split(q: f32) -> (f32, f32) {
        let hi = half_to_f32(f32_to_f16_bits(q));
        let lo = half_to_f32(f32_to_f16_bits(q - hi));
        (hi, lo)
    }

    #[derive(Clone, Copy, PartialEq)]
    enum ScoreMut {
        Base,
        NoTurn,
        Scale128,
        NextHead,
    }

    /// One token's f64 rule over its pools, and the band per pool.
    struct Rule {
        score: Vec<f64>,
        band: Vec<f64>,
    }

    /// Token `t` of `tokens` (the gemvs' layout: value `i` of the query at `i·tokens + t`) at
    /// `heads` heads, live count `c`, over the first `n` pools of `keys`: the f64 rule (the
    /// query turned in f64 at position `c − 1` by `table`'s f32 cos and sin, the weights scaled
    /// in f32 as the kernel does, `relu`, the sum exact) and its band — `gate_glm_sel`'s
    /// tensor-core band over the kernel's own turned and split query, its split's distance to
    /// its turned f32 query, and that query's distance to the f64 turn (two products and an add
    /// a value), each carried by `|w_h|`, then the head sum's roundings.
    #[allow(
        clippy::too_many_arguments,
        reason = "the rule's inputs as the launch takes them, and the mutant it runs"
    )]
    fn rule(
        q: &[f32],
        w: &[f32],
        (tokens, t, heads): (usize, usize, usize),
        c: usize,
        table: &[f32],
        keys: &[u16],
        n: usize,
        scale: f32,
        mu: ScoreMut,
    ) -> Rule {
        let tab = &table[(c - 1) * ROT..][..ROT];
        // Per value: the kernel's turned f32 query's split, the f64 turned query, and the
        // turn's f32 distance bound.
        let mut sp = vec![[0.0f64; 4]; heads * DIM];
        for h in 0..heads {
            let at = |i: usize| q[(h * DIM + i) * tokens + t];
            for i in 0..DIM {
                sp[h * DIM + i] = [0.0, 0.0, f64::from(at(i)), 0.0];
            }
            for i in 0..ROT / 2 {
                let d = DIM - ROT + 2 * i;
                let (x0, x1) = (at(d), at(d + 1));
                let (cs, sn) = (tab[2 * i], tab[2 * i + 1]);
                if mu != ScoreMut::NoTurn {
                    let (c64, s64) = (f64::from(cs), f64::from(sn));
                    let (a, b) = (f64::from(x0), f64::from(x1));
                    sp[h * DIM + d][2] = a * c64 - b * s64;
                    sp[h * DIM + d + 1][2] = a * s64 + b * c64;
                    let m = (a * c64).abs() + (b * s64).abs();
                    let m1 = (a * s64).abs() + (b * c64).abs();
                    sp[h * DIM + d][3] = 2.0 * U * m;
                    sp[h * DIM + d + 1][3] = 2.0 * U * m1;
                }
            }
            // The kernel's own f32 turn, then its split.
            for i in 0..DIM {
                let y = if i >= DIM - ROT {
                    let pi = (i - (DIM - ROT)) / 2;
                    let d = DIM - ROT + 2 * pi;
                    let (y0, y1) = turn(at(d), at(d + 1), tab[2 * pi], tab[2 * pi + 1]);
                    if i == d { y0 } else { y1 }
                } else {
                    at(i)
                };
                let (hi, lo) = split(y);
                sp[h * DIM + i][0] = f64::from(hi);
                sp[h * DIM + i][1] = f64::from(lo);
                // The split's distance to the kernel's f32 query joins the turn's.
                sp[h * DIM + i][3] += (f64::from(y) - f64::from(hi) - f64::from(lo)).abs();
            }
        }
        let sc = if mu == ScoreMut::Scale128 {
            1.0 / 128f32.sqrt()
        } else {
            scale
        };
        let w64: Vec<f64> = (0..heads)
            .map(|h| {
                let hh = if mu == ScoreMut::NextHead {
                    (h + 1) % heads
                } else {
                    h
                };
                f64::from(w[hh * tokens + t] * sc)
            })
            .collect();
        let mut out = Rule {
            score: Vec::with_capacity(n),
            band: Vec::with_capacity(n),
        };
        let hs = gamma(HEAD_SUM_ROUNDINGS);
        for j in 0..n {
            let k: Vec<f64> = keys[j * DIM..(j + 1) * DIM]
                .iter()
                .map(|&b| widen(b))
                .collect();
            let (mut s, mut kb, mut smag) = (0.0f64, 0.0f64, 0.0f64);
            for (h, &wh) in w64.iter().enumerate() {
                let (mut exact, mut tc, mut p, mut rep) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
                for (&[hi, lo, q64, e], &kv) in sp[h * DIM..(h + 1) * DIM].iter().zip(&k) {
                    exact += q64 * kv;
                    tc += (hi + lo) * kv;
                    p += (hi * kv).abs() + (lo * kv).abs();
                    rep += e * kv.abs();
                }
                let act = exact.max(0.0);
                s += wh * act;
                smag += wh.abs() * act.abs();
                kb += wh.abs() * (TC_UNITS * U * p + 2.0 * U * tc.abs() + rep);
            }
            out.score.push(s);
            out.band.push(MARGIN * ((1.0 + hs) * kb + hs * smag));
        }
        out
    }

    /// The score case's counts: one row that sees 511 pools and is not scored, then four that are
    /// (tails of one, four, four and one).
    const SCORE_COUNTS: [u32; 5] = [2048, 2049, 9000, 9001, 16_384];

    /// Clause 4; returns the card's scores and each scored row's f64 rule for clause 6.
    fn score_case(cx: &Ctx<'_>) -> Result<(bool, Scored), GateError> {
        let tokens = SCORE_COUNTS.len();
        let pools = pools_for(CTX);
        let mut r = Lcg(0x7363_6f72);
        let plane = r.f16(pools * DIM, -1.0, 1.0);
        let q = r.fill(IDX_HEADS * DIM * tokens, -1.0, 1.0);
        let w = r.fill(IDX_HEADS * tokens, -1.0, 1.0);
        let table = rope_rows(CTX, false);
        let scale = weights_scale(IDX_HEADS, DIM);
        let sink = cx.gpu.unlabelled_sink();
        let sc = cx.score(
            sink,
            &q,
            &w,
            &SCORE_COUNTS,
            &table,
            &plane,
            IDX_HEADS,
            scale,
        )?;
        let again = cx.score(
            sink,
            &q,
            &w,
            &SCORE_COUNTS,
            &table,
            &plane,
            IDX_HEADS,
            scale,
        )?;
        let rerun = bits_equal(&sc, &again);
        let mut alone = true;
        for t in 0..tokens {
            let qt: Vec<f32> = (0..IDX_HEADS * DIM).map(|i| q[i * tokens + t]).collect();
            let wt: Vec<f32> = (0..IDX_HEADS).map(|h| w[h * tokens + t]).collect();
            let one = cx.score(
                sink,
                &qt,
                &wt,
                &SCORE_COUNTS[t..=t],
                &table,
                &plane,
                IDX_HEADS,
                scale,
            )?;
            alone &= bits_equal(&one, &sc[t * pools..(t + 1) * pools]);
        }
        let word = cx.gpu.take_fault()?;
        let (mut worst, mut poison_ok, mut glm_slot_nan) = (0.0f64, true, false);
        let mut killed = [false; 3];
        let mut rules = Vec::new();
        for (t, &c) in SCORE_COUNTS.iter().enumerate() {
            let row = &sc[t * pools..(t + 1) * pools];
            let n = if scored_tail(c, CTX as u32, KEPT as u32) {
                (c as usize - 1) / POOL
            } else {
                0
            };
            poison_ok &= row[n..].iter().all(|v| v.is_nan());
            // GLM-5.3's rule scores `c / 4` pools from 512 kept: a row it scores past this one's
            // pools names a slot the pass leaves NaN.
            let glm_n = if c as usize / POOL > KEPT {
                c as usize / POOL
            } else {
                0
            };
            glm_slot_nan |= glm_n > n && row[glm_n - 1].is_nan();
            if n == 0 {
                rules.push(None);
                continue;
            }
            let shape = (tokens, t, IDX_HEADS);
            let ru = rule(
                &q,
                &w,
                shape,
                c as usize,
                &table,
                &plane,
                n,
                scale,
                ScoreMut::Base,
            );
            for (j, &rj) in row.iter().enumerate().take(n) {
                let d = (f64::from(rj) - ru.score[j]).abs();
                worst = worst.max(if d.is_nan() {
                    f64::INFINITY
                } else {
                    d / ru.band[j]
                });
            }
            for (mi, mu) in [ScoreMut::NoTurn, ScoreMut::Scale128, ScoreMut::NextHead]
                .into_iter()
                .enumerate()
            {
                let m = rule(&q, &w, shape, c as usize, &table, &plane, n, scale, mu);
                killed[mi] |= (0..n).any(|j| (m.score[j] - f64::from(row[j])).abs() > ru.band[j]);
            }
            rules.push(Some(ru));
        }
        let pass = worst <= 1.0
            && poison_ok
            && rerun
            && alone
            && word.is_none()
            && killed == [true; 3]
            && glm_slot_nan;
        println!(
            "score counts {SCORE_COUNTS:?} at {IDX_HEADS} heads over {pools} pools: worst |card − \
             f64 rule| / band {worst:.3} (<= 1); slots from each row's (c − 1)/4 pools on NaN \
             {poison_ok}; rerun bit for bit {rerun}; each token alone its column bit for bit \
             {alone}; fault={}; mutants past the band (no turn, scale 1/√128, head h+1's weight) \
             {killed:?}; GLM's c/4 pools name a NaN slot {glm_slot_nan} {}",
            fault_text(word),
            verdict(pass)
        );
        Ok((pass, Scored { scores: sc, rules }))
    }

    /// Clause 4's card scores and each row's f64 rule (`None` for the unscored row).
    struct Scored {
        scores: Vec<f32>,
        rules: Vec<Option<Rule>>,
    }

    // ----------------------------------------------------------- 5. heads

    /// Clause 5.
    fn heads_case(cx: &Ctx<'_>) -> Result<bool, GateError> {
        let counts = [9001u32, 16_384];
        let tokens = counts.len();
        let pools = pools_for(CTX);
        let mut r = Lcg(0x6865_6164);
        let plane = r.f16(pools * DIM, -1.0, 1.0);
        let q = r.fill(HEADS * DIM * tokens, -1.0, 1.0);
        let w = r.fill(HEADS * tokens, -1.0, 1.0);
        let ident = rope_rows(CTX, true);
        let scale = weights_scale(HEADS, DIM);
        let ours = cx.score(
            cx.gpu.unlabelled_sink(),
            &q,
            &w,
            &counts,
            &ident,
            &plane,
            MAX_HEADS,
            scale,
        )?;
        let glm = cx.score_glm(&q, &w, &counts, &plane, scale)?;
        let word = cx.gpu.take_fault()?;
        let mut same = true;
        let mut compared = 0usize;
        for (t, &c) in counts.iter().enumerate() {
            let n = (c as usize - 1) / POOL;
            same &= bits_equal(
                &ours[t * pools..t * pools + n],
                &glm[t * pools..t * pools + n],
            );
            compared += n;
        }
        let pass = same && compared > 0 && word.is_none();
        println!(
            "heads {MAX_HEADS} under the identity turn, counts {counts:?}: {compared} pools bit for \
             bit GLM-5.3's kpool_score {same}; fault={} {}",
            fault_text(word),
            verdict(pass)
        );
        Ok(pass)
    }

    // ------------------------------------------------------------ 6. top-k

    /// Row `c`'s list by the rule: the `kept` best of its `(c − 1) / 4` scores by
    /// [`order_key`], `tie` at the cut, their tokens in pool order, then the tail `4n .. c`;
    /// `0 .. c` for a row that sees at most `kept` pools; empty for a refused count.
    fn rule_list(c: usize, scores: &[f32], kept: usize, tie: Tie) -> Vec<u32> {
        if c == 0 || c > CTX {
            return Vec::new();
        }
        let n = (c - 1) / POOL;
        if n <= kept {
            return (0..c as u32).collect();
        }
        let pools = select_pools_by(&scores[..n], kept, tie);
        let mut out: Vec<u32> = pools
            .iter()
            .flat_map(|&j| (0..POOL as u32).map(move |i| POOL as u32 * j + i))
            .collect();
        out.extend((POOL * n) as u32..c as u32);
        out
    }

    /// GLM-5.3's list at `kept` pools: `c / 4` pools seen, its tail `4·(c / 4) .. c`.
    fn glm_list(c: usize, scores: &[f32], kept: usize) -> Vec<u32> {
        let n = c / POOL;
        if n <= kept {
            return (0..c as u32).collect();
        }
        let pools = select_pools_by(&scores[..n], kept, Tie::Lower);
        let mut out: Vec<u32> = pools
            .iter()
            .flat_map(|&j| (0..POOL as u32).map(move |i| POOL as u32 * j + i))
            .collect();
        out.extend((POOL * n) as u32..c as u32);
        out
    }

    /// Clause 6 over clause 4's scores, planted rows of ties and rows that do not select.
    fn topk_case(cx: &Ctx<'_>, scored: &Scored) -> Result<bool, GateError> {
        let pools = pools_for(CTX);
        // The planted rows at counts 8,193 and 8,196 (2,048 pools seen): 293 distinct positive
        // scores, 136 more equal at 0.25, and relu zeros of both signs for the other 1,619, so
        // the 511th key is a zero and the cut takes 82 of the 1,619 tied pools.
        let mut ties = vec![0.0f32; pools];
        for (j, v) in ties.iter_mut().enumerate().take(2048) {
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
        let mut scores = scored.scores.clone();
        for c in [8193u32, 8196] {
            counts.push(c);
            scores.extend(&ties);
        }
        for c in [1u32, 4, 5, 2048] {
            counts.push(c);
            scores.extend(std::iter::repeat_n(f32::NAN, pools));
        }
        let (list, lens) = cx.topk(cx.gpu.unlabelled_sink(), &counts, &scores)?;
        let word = cx.gpu.take_fault()?;
        let mut ok = word.is_none();
        let mut mism = Vec::new();
        let mut differs = [false; 3];
        for (t, &c) in counts.iter().enumerate() {
            let c = c as usize;
            let row = &scores[t * pools..(t + 1) * pools];
            let want = rule_list(c, row, KEPT, Tie::Lower);
            let len = lens[t] as usize;
            let got = &list[t * WIDTH..t * WIDTH + len.min(WIDTH)];
            let rest_ok = list[t * WIDTH + len.min(WIDTH)..(t + 1) * WIDTH]
                .iter()
                .all(|&v| v == LIST_POISON);
            if got != want.as_slice() || len != want.len() || !rest_ok {
                mism.push(c);
                ok = false;
            }
            if (c - 1) / POOL > KEPT {
                differs[0] |= rule_list(c, row, KEPT, Tie::Higher) != got;
                differs[1] |= rule_list(c, row, KEPT + 1, Tie::Lower) != got;
                differs[2] |= glm_list(c, row, KEPT) != got;
            }
        }
        // As a set: each scored row's kept pools against the f64 rule's best.
        let mut set_ok = true;
        let mut set_lines = Vec::new();
        for (t, ru) in scored.rules.iter().enumerate() {
            let Some(ru) = ru else { continue };
            let n = ru.score.len();
            let card: Vec<usize> = (0..KEPT)
                .map(|i| list[t * WIDTH + POOL * i] as usize / POOL)
                .collect();
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_by(|&a, &b| ru.score[b].total_cmp(&ru.score[a]).then(a.cmp(&b)));
            let cut = order[KEPT - 1];
            let mut exact: Vec<usize> = order[..KEPT].to_vec();
            exact.sort_unstable();
            let only_card: Vec<usize> = card
                .iter()
                .copied()
                .filter(|j| exact.binary_search(j).is_err())
                .collect();
            let only_exact: Vec<usize> = exact
                .iter()
                .copied()
                .filter(|j| card.binary_search(j).is_err())
                .collect();
            let within = only_card
                .iter()
                .chain(&only_exact)
                .all(|&j| (ru.score[j] - ru.score[cut]).abs() <= ru.band[j] + ru.band[cut]);
            set_ok &= within;
            set_lines.push(format!(
                "c={}: {} pools differ, within the cut's bands {within}",
                counts[t],
                only_card.len() + only_exact.len()
            ));
        }
        // The refused counts, each in its own layer.
        let mut refusal = Vec::new();
        for (i, c) in [0u32, CTX as u32 + 1].into_iter().enumerate() {
            let layer = 50 + i;
            let sink = cx.gpu.layer_sink(layer)?;
            let (_, lens) = cx.topk(sink, &[c], &vec![0.0; pools])?;
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
        let pass = ok && set_ok && differs == [true; 3];
        println!(
            "topk counts {counts:?}: every list, its length and the slots past it the rule's (the \
             lower pool of a tie, the forced tail) — mismatched at {mism:?}; fault={}; as sets \
             against the f64 rule [{}]; refused [{}]; mutants differ (higher tie, 512 kept, GLM's \
             c/4 pools) {differs:?} {}",
            fault_text(word),
            set_lines.join("; "),
            refusal.join("; "),
            verdict(pass)
        );
        Ok(pass)
    }

    // ---------------------------------------------------------- 7. faults

    /// Clause 7: each plant in its own layer, then a clean launch.
    fn fault_cases(cx: &Ctx<'_>) -> Result<Vec<bool>, GateError> {
        let mut out = Vec::new();
        let mut r = Lcg(0x6661_756c);
        let (m, ctx) = (8usize, 64usize);
        let x = r.fill(m * INDEX_HEAD, -2.0, 2.0);
        let gain = r.fill(INDEX_HEAD, 0.5, 1.5);
        let (wv, wg) = (Q8::random(&mut r), Q8::random(&mut r));
        let pos: Vec<u32> = (0..m as u32).collect();
        let mut nan_x = x.clone();
        nan_x[3 * INDEX_HEAD + 9] = f32::NAN;
        let mut over_pos = pos.clone();
        over_pos[5] = ctx as u32;
        let init = vec![NAN_F16; ctx * INDEX_ROW];
        let mut cache = r.f16(ctx * INDEX_ROW, -2.0, 2.0);
        let clean_cache = cache.clone();
        cache[5 * INDEX_ROW + 7] = NAN_F16;
        let ape = r.fill(POOL * INDEX_HEAD, -1.0, 1.0);
        let pgain = r.fill(INDEX_HEAD, 0.5, 1.5);
        let table = rope_rows(CTX, false);
        let p = PoolIn {
            ape: &ape,
            gain: &pgain,
            table: &table[..ctx * ROT],
            ctx,
        };
        let plane = vec![NAN_F16; pools_for(ctx) * INDEX_HEAD];
        let counts: Vec<u32> = (1..=ctx as u32).collect();
        let over = [ctx as u32 + 1];
        let pools = pools_for(CTX);
        let splane = r.f16(pools * DIM, -1.0, 1.0);
        let q = r.fill(IDX_HEADS * DIM, -1.0, 1.0);
        let w = r.fill(IDX_HEADS, -1.0, 1.0);
        let scale = weights_scale(IDX_HEADS, DIM);
        let mut nan_q = q.clone();
        nan_q[3] = f32::NAN;
        type Run<'a> = Box<dyn Fn(&Ctx<'_>, FaultSink, bool) -> Result<(), GateError> + 'a>;
        let plants: [(usize, &str, FaultSite, Run<'_>); 5] = [
            (
                60,
                "a NaN raw key",
                FaultSite::CacheValue,
                Box::new(|cx, sink, planted| {
                    let a = AppendIn {
                        x: if planted { &nan_x } else { &x },
                        stride: INDEX_HEAD,
                        k_off: 0,
                        gain: &gain,
                        wv: &wv,
                        wg: &wg,
                        pos: &pos,
                        ctx,
                    };
                    cx.append(sink, &a, &init).map(|_| ())
                }),
            ),
            (
                61,
                "an append position at the context",
                FaultSite::CachePos,
                Box::new(|cx, sink, planted| {
                    let a = AppendIn {
                        x: &x,
                        stride: INDEX_HEAD,
                        k_off: 0,
                        gain: &gain,
                        wv: &wv,
                        wg: &wg,
                        pos: if planted { &over_pos } else { &pos },
                        ctx,
                    };
                    cx.append(sink, &a, &init).map(|_| ())
                }),
            ),
            (
                62,
                "a NaN pool member",
                FaultSite::PoolSelect,
                Box::new(|cx, sink, planted| {
                    let c = if planted { &cache } else { &clean_cache };
                    cx.pool(sink, c, &p, &counts, &plane).map(|_| ())
                }),
            ),
            (
                63,
                "a pool count past the context",
                FaultSite::CachePos,
                Box::new(|cx, sink, planted| {
                    let c: &[u32] = if planted { &over } else { &counts };
                    cx.pool(sink, &clean_cache, &p, c, &plane).map(|_| ())
                }),
            ),
            (
                64,
                "a NaN indexer query",
                FaultSite::PoolSelect,
                Box::new(|cx, sink, planted| {
                    let qq = if planted { &nan_q } else { &q };
                    cx.score(sink, qq, &w, &[9001], &table, &splane, IDX_HEADS, scale)
                        .map(|_| ())
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

    // --------------------------------------------------------- 8. refuse

    /// Whether `r` is the named shape refusal of `what`.
    fn refused_by(r: &Result<(), GpuError>, what: &str) -> bool {
        matches!(r, Err(GpuError::Shape { what: w, .. }) if *w == what)
    }

    /// Clause 8.
    fn refuse_case(cx: &Ctx<'_>) -> Result<bool, GateError> {
        let s = cx.stream();
        let pools = pools_for(CTX);
        let mut lines = Vec::new();
        let mut all = true;
        let n_keys = DeviceBuffer::from_host(s, &[9001u32])?;
        let table = DeviceBuffer::from_host(s, &rope_rows(CTX, false))?;
        let pooled = DeviceTensor::upload(s, &vec![0u16; pools * DIM], pools, DIM)?;
        for heads in [12usize, 40] {
            let q = DeviceBuffer::from_host(s, &vec![0.0f32; heads * DIM])?;
            let w = DeviceBuffer::from_host(s, &vec![0.0f32; heads])?;
            let mut scores = DeviceBuffer::from_host(s, &vec![0.0f32; pools])?;
            let r = cx.kk.enqueue_score_tail(
                s,
                TailScoreArgs {
                    q: &q,
                    w: &w,
                    n_keys: &n_keys,
                    table: &table,
                    pooled: &pooled,
                    heads,
                    tokens: 1,
                    ctx: CTX,
                    kept: KEPT,
                    scale: 1.0,
                    fault: cx.gpu.unlabelled_sink(),
                    scores: &mut scores,
                },
            );
            let ok = refused_by(&r, "kpool::enqueue_score_tail");
            all &= ok;
            lines.push(format!("score at {heads} heads refused {ok}"));
        }
        {
            let sc = DeviceBuffer::from_host(s, &vec![0.0f32; pools])?;
            let mut list = DeviceBuffer::from_host(s, &vec![0u32; WIDTH])?;
            let mut n_sel = DeviceBuffer::from_host(s, &[0u32])?;
            let r = cx.qk.enqueue_topk_tail(
                s,
                TopkTailArgs {
                    n_keys: &n_keys,
                    scores: &sc,
                    ctx: CTX,
                    kept: KEPT,
                    width: WIDTH - 1,
                    m: 1,
                    fault: cx.gpu.unlabelled_sink(),
                    list: &mut list,
                    n_sel: &mut n_sel,
                },
            );
            let ok = refused_by(&r, "qsa::enqueue_topk_tail");
            all &= ok;
            lines.push(format!("lists of {} entries refused {ok}", WIDTH - 1));
        }
        {
            let cache = DeviceTensor::upload(s, &vec![0u16; 8 * INDEX_ROW], 8, INDEX_ROW)?;
            let ape = DeviceBuffer::from_host(s, &vec![0.0f32; POOL * INDEX_HEAD])?;
            let gain = DeviceBuffer::from_host(s, &vec![1.0f32; INDEX_HEAD])?;
            let nk = DeviceBuffer::from_host(s, &[4u32; 6])?;
            let mut plane = DeviceTensor::upload(s, &vec![0u16; 16 * INDEX_HEAD], 16, INDEX_HEAD)?;
            let r = cx.lk.enqueue_index_pool_norm_rope(
                s,
                IndexPoolRopeArgs {
                    cache: &cache,
                    ape: &ape,
                    gain: &gain,
                    table: &table,
                    n_keys: &nk,
                    eps: EPS,
                    ctx: 64,
                    m: 6,
                    fault: cx.gpu.unlabelled_sink(),
                    pooled: &mut plane,
                },
            );
            let ok = refused_by(&r, "enqueue_index_pool_norm_rope");
            all &= ok;
            lines.push(format!("6 counts over a ring of 8 refused {ok}"));
        }
        {
            let x = DeviceBuffer::from_host(s, &vec![0.0f32; INDEX_HEAD])?;
            let gain = DeviceBuffer::from_host(s, &vec![1.0f32; INDEX_HEAD])?;
            let pos = DeviceBuffer::from_host(s, &[0u32])?;
            let good = Q8Dev::upload(s, &Q8::random(&mut Lcg(1)))?;
            let short = Q8Dev {
                qs: DeviceTensor::upload(s, &vec![0u32; 64 * 32], 64, 32)?,
                d: DeviceTensor::upload(s, &vec![0u16; 64 * 4], 64, 4)?,
            };
            let mut cache = DeviceTensor::upload(s, &vec![0u16; 4 * INDEX_ROW], 4, INDEX_ROW)?;
            let r = cx.lk.enqueue_index_key_rms_append(
                s,
                IndexKeyRmsArgs {
                    rows: Rows {
                        x: &x,
                        stride: INDEX_HEAD,
                        m: 1,
                    },
                    k_off: 0,
                    gain: &gain,
                    value: good.proj(),
                    gate: short.proj(),
                    pos: &pos,
                    eps: EPS,
                    ctx: 4,
                    fault: cx.gpu.unlabelled_sink(),
                    cache: &mut cache,
                },
            );
            let ok = refused_by(&r, "enqueue_index_key_rms_append");
            all &= ok;
            lines.push(format!("a 64-row gate projection refused {ok}"));
        }
        let word = cx.gpu.take_fault()?;
        let pass = all && word.is_none();
        println!(
            "refuse: {}; fault={} {}",
            lines.join("; "),
            fault_text(word),
            verdict(pass)
        );
        Ok(pass)
    }
}
