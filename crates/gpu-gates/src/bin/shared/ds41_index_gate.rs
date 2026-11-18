//! GPU gate for DeepSeek-V4.1's lightning indexer — B4 op block F
//! (`docs/research/v41-b4-plan-report.md` §1-F) — `bloomery_gpu::indexer`
//! against our own rules, on the model file's own projection weights over
//! seeded inputs. Every emitting layer (`LayerKind::indexer`) at the decode
//! steps of position 301 (top_k 64) and 1,025 (the file's top_k) gets a
//! launch of the op on the engine's path and a rerun: our gemv of seeded
//! activations in the file's type of `attn_q_b` (q8_0 on f32 activations, or
//! q3_K on q8_1 activations) and our q3_K gemv of seeded activations through
//! `indexer.proj` (q8_1 activations) feed the indexer, over a seeded key cache
//! (NaN past the visible rows).
//!
//! Per layer, the B4 gate form:
//! - (i) the query after rope and Hadamard, and the scaled weights, against
//!   this binary's transcription of our rope and transform on the gemv's own
//!   output, and of the scale, bit for bit; our rope table against ggml's
//!   recipe; a q3_K query gemv also by itself, each row against the exact dot
//!   of our q8_1 values within `KERNEL_BAND`.
//! - (ii) the scores against our rule ([`rule`]) within the tensor-core band.
//! - (iii) the ids: the list against the rule's selection at every row outside
//!   the tie band of the k-th row ([`ids_vs_rule`]); the list exactly the
//!   top-k of the kernel's own scores (order-preserving keys, ties to the
//!   lower row), `k` entries in ascending order with the rest of the row
//!   untouched.
//! - (iv) the identity: at position 301 over the file's top_k (at most 302
//!   visible rows) the list is `0..n_vis` and the score pass writes nothing;
//!   the selected-row attention over such a list is bit-identical to the
//!   prefix attention.
//! - (v) depth cases the file's own steps cannot reach: synthetic keys, query
//!   and weights at 16,384 and 32,768 visible rows of a longer cache, with a
//!   group of duplicated key rows planted to straddle the threshold
//!   (bit-identical scores, so the lower rows must win), against our rule and
//!   its selection; and a clustered case whose scores share their top bits,
//!   against the exact top-k of the kernel's scores.
//! - (vi) reruns bit-identical; one captured graph replayed over rewritten
//!   counts (`n_vis` and `top_k`, selecting and identity), each replay
//!   bit-identical to an eager run; no local depot in either entry.
//!
//! Every output is poisoned before a launch (NaN query, weights and scores,
//! `u32::MAX` list entries), so a slot written where it should not be, or
//! not written, shows, and the histogram must be zero after every pair.

use std::collections::HashMap;
use std::f32::consts::FRAC_1_SQRT_2;

use bloomery_gpu::indexer::{HEAD_DIM, HEADS, IndexerArgs, IndexerKernels, IndexerScratch};
use bloomery_gpu::weights::{DevWeight, q8_0_planes, upload_file_tensor};
use bloomery_gpu::{DeviceTensor, Gpu, GpuError, Q8Act};
use bloomery_gpu_deepseek41::attn::{self, AttnArgs, AttnKernels, LATENT, SelectedRows};
use bloomery_gpu_deepseek41::params::rope_specs;
use bloomery_gpu_deepseek41::rope::{Direction, RopeSpec, RopeTable, ggml_rope_cache};
use bloomery_gpu_gates::act_rule::{self, Q3kWeight};
use bloomery_gpu_gates::ik_q8_2::QK;
use bloomery_gpu_gates::{
    GateError, KERNEL_BAND, NAN_F16, activations, bits_equal, checks_failed, max_rel_err,
    no_local_depot, ref_model_path, verdict,
};
use cuda_core::{CudaStream, DeviceBuffer};
use gguf::Split;
use gguf::quant::{GgmlType, Q8Block, f32_to_f16_bits, half_to_f32};
use model::arch::Arch;
use model::arch::deepseek41::hparams::Hparams;
use model::arch::deepseek41::plan::{Planner, StepPlan};

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

// ------------------------------------------------------------ weights

/// A q8_0 weight: its q8f32 planes for the gemv (the loader's own
/// `q8_0_planes`).
struct Q8 {
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

// ------------------------------------------------------ score rules

/// The split of one query value the kernel makes: `hi = f16(q)`,
/// `lo = f16(q − hi)`, each widened back.
fn split(q: f32) -> (f32, f32) {
    let hi = half_to_f32(f32_to_f16_bits(q));
    let lo = half_to_f32(f32_to_f16_bits(q - hi));
    (hi, lo)
}

/// Our rule over one token's rows, and the kernel's band per row.
struct Rule {
    /// `Σ_h w_h · relu(Σ_d (hi + lo)·k)`: the kernel's split query, its
    /// weights, the exact f16 products summed in f64.
    score: Vec<f64>,
    /// The kernel's distance to `score`: per head the tensor cores'
    /// [`TC_UNITS`] of the products' magnitude and one rounding of
    /// `acc_hi + acc_lo`, carried by `|w_h|` (relu moves nothing further);
    /// then the head sum's [`HEAD_SUM_ROUNDINGS`].
    kernel: Vec<f64>,
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
    let mut rows = vec![[0.0f64; 2]; n];
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
                    let (mut sc, mut kb, mut smag) = (0.0f64, 0.0f64, 0.0f64);
                    for (h, &wh) in w64.iter().enumerate() {
                        let (mut dot, mut p) = (0.0f64, 0.0f64);
                        for (&[hi, lo, _], &kv) in
                            sp[h * HEAD_DIM..(h + 1) * HEAD_DIM].iter().zip(&k)
                        {
                            dot += (hi + lo) * kv;
                            p += (hi * kv).abs() + (lo * kv).abs();
                        }
                        let relu = dot.max(0.0);
                        sc += wh * relu;
                        smag += wh.abs() * relu;
                        kb += wh.abs() * (TC_UNITS * U * p + 2.0 * U * dot.abs());
                    }
                    let hs = gamma(HEAD_SUM_ROUNDINGS);
                    *o = [sc, (1.0 + hs) * kb + hs * smag];
                }
            });
        }
    });
    Rule {
        score: rows.iter().map(|r| r[0]).collect(),
        kernel: rows.iter().map(|r| r[1]).collect(),
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

    fn enqueue(&mut self, kernels: &IndexerKernels, stream: &CudaStream) -> Result<(), GpuError> {
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
    let kernels = bloomery_gpu_deepseek41::indexer::kernels(gpu.context(), &hp)?;
    let attn = AttnKernels::load(gpu.context())?;
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
        split,
        hp,
        yarn,
        q8: HashMap::new(),
        q3: HashMap::new(),
    };
    let mut ok = no_local_depot(&["ds41_indexer_score", "ds41_indexer_topk"])?;
    let mut layers = 0usize;
    let mut failed = 0usize;
    for case in &SELECT_STEPS {
        let (n, f) = gate_step(&mut cx, case)?;
        layers += n;
        failed += f;
    }
    ok &= identity_step(&mut cx, &IDENTITY_STEP)?;
    ok &= identity_attention(&cx)?;
    for (n, seed) in [(16_384usize, 11u32), (32_768, 12)] {
        ok &= planted_case(&cx, n, seed)?;
    }
    ok &= clustered_case(&cx, 32_768, 13)?;
    ok &= replay(&cx)?;
    // PIN(2026-11-18): the real-file ik set retired; ik cannot read bq3km's engram schema; no fixture set carries this clause
    ok &= failed == 0;
    println!(
        "gate_deepseek41_index: {layers} layer cases, {failed} failed; identity, attention, depth \
             and replay checks above — {}",
        verdict(ok)
    );
    if ok { Ok(()) } else { Err(checks_failed()) }
}

// ----------------------------------------------------------- the steps

/// A decode step the per-layer clauses run at: the position, the cache's
/// context and the selection width, the file's unless set.
struct StepCase {
    label: &'static str,
    pos: u32,
    ctx: u64,
    top_k: Option<usize>,
}

/// The steps whose indexer layers select: position 301 at a top_k of 64 and
/// position 1,025 at the file's top_k.
const SELECT_STEPS: [StepCase; 2] = [
    StepCase {
        label: "pos301-top64",
        pos: 301,
        ctx: 512,
        top_k: Some(64),
    },
    StepCase {
        label: "pos1025",
        pos: 1025,
        ctx: 2048,
        top_k: None,
    },
];
/// The step where none selects: position 301 at the file's top_k, over at
/// most 302 visible rows.
const IDENTITY_STEP: StepCase = StepCase {
    label: "pos301",
    pos: 301,
    ctx: 512,
    top_k: None,
};

/// One step's position, selection width and plan.
struct Step {
    pos: u32,
    top_k: usize,
    ctx: u64,
    planner: Planner,
    plan: StepPlan,
}

fn step_of(cx: &Cx, case: &StepCase) -> Result<Step, GateError> {
    let top_k = case.top_k.unwrap_or(cx.hp.indexer.top_k);
    let hp = cx.hp.clone().with_indexer_top_k(top_k);
    let planner = Planner::from_file(&cx.split, &hp, case.ctx)?;
    // One token after a prefill of zeros: the plan's visible counts do not
    // read the ids.
    let before = vec![0u32; planner.ngram()];
    let mut plan = StepPlan::default();
    planner.plan_into(&[0], case.pos, &before, &mut plan)?;
    Ok(Step {
        pos: case.pos,
        top_k: hp.indexer.top_k,
        ctx: case.ctx,
        planner,
        plan,
    })
}

/// A layer's visible rows and its stream's cache height.
fn layer_rows(step: &Step, l: usize) -> Result<(usize, usize), GateError> {
    let s = step
        .planner
        .layer_stream(l)
        .ok_or_else(|| format!("layer {l} runs the indexer and has no stream"))?;
    let st = &step.plan.streams[s];
    let n_vis = st.n_visible[0] as usize;
    let rows = usize::try_from(step.ctx)?.div_ceil(st.ratio as usize);
    Ok((n_vis, rows))
}

/// Gate every emitting layer of one selecting step: (layer cases, failed).
fn gate_step(cx: &mut Cx, case: &StepCase) -> Result<(usize, usize), GateError> {
    let step = step_of(cx, case)?;
    println!(
        "step {}: position {}, top_k {}, context {}",
        case.label, step.pos, step.top_k, step.ctx
    );
    let (mut n, mut failed) = (0usize, 0usize);
    for l in 0..cx.hp.layers.len() {
        if cx.hp.layers[l].indexer {
            n += 1;
            failed += usize::from(!gate_layer(cx, case.label, &step, l)?);
        }
    }
    Ok((n, failed))
}

fn gate_layer(cx: &mut Cx, label: &str, step: &Step, l: usize) -> Result<bool, GateError> {
    let (n_vis, rows) = layer_rows(step, l)?;
    let k = step.top_k;
    if n_vis <= k {
        return Err(
            format!("{label} layer {l}: {n_vis} visible rows select nothing at top_k {k}").into(),
        );
    }
    let q_name = format!("blk.{l}.indexer.attn_q_b.weight");
    let w_name = format!("blk.{l}.indexer.proj.weight");
    cx.ensure_q8_or_q3(&q_name)?;
    cx.ensure_q3(&w_name)?;
    let cx = &*cx;
    let nq = HEADS * HEAD_DIM;
    // The seeded inputs: the two projections' activations, and the key
    // source, NaN above the visible rows.
    let seed = 4 * (step.pos + u32::try_from(l)?);
    let q_k = cx
        .q8
        .get(&q_name)
        .map_or_else(|| cx.q3[&q_name].host.k, |w| w.k);
    let x_q = activations(q_k, 1, seed);
    let x_w = activations(cx.q3[&w_name].host.k, 1, seed + 1);
    let mut keys: Vec<u16> = activations(HEAD_DIM, rows, seed + 2)
        .into_iter()
        .map(f32_to_f16_bits)
        .collect();
    keys[n_vis * HEAD_DIM..].fill(NAN_F16);
    let (cs, cs_ik) = cx.tables(step.pos)?;
    let tables_same = bits_equal(&cs, &cs_ik);
    let scale = cx.kernels.scale();

    // The gemvs: the query's in the file's type of attn_q_b. A q3_K query
    // gemv is also pinned by itself, per row, against the exact dot of our
    // q8_1 values within KERNEL_BAND (gate_p1's form).
    let (q_raw, q_gemv_rel) = if let Some(q8) = cx.q8.get(&q_name) {
        let stream = cx.gpu.stream();
        let x_dev = DeviceBuffer::from_host(stream, &x_q)?;
        let mut y = DeviceBuffer::<f32>::zeroed(stream, q8.rows)?;
        cx.gpu
            .q8f32()
            .enqueue_q8_0_gemv(stream, &q8.qs, &q8.d, &x_dev, 1, &mut y)?;
        stream.synchronize()?;
        (y.to_host_vec(stream)?, None)
    } else {
        let q3 = &cx.q3[&q_name];
        let rows = act_rule::q3k_rows(&q3.host, &x_q, q3.host.rows);
        let y = q3k_gemv(cx, q3, &x_q)?;
        let ours: Vec<f32> = rows.iter().map(|r| r.ours as f32).collect();
        let rel = max_rel_err(&y, &ours).unwrap_or(f32::INFINITY);
        (y, Some(rel))
    };
    let q_gemv_ok = q_gemv_rel.is_none_or(|rel| rel <= KERNEL_BAND);
    let q_gemv_line =
        q_gemv_rel.map_or_else(String::new, |rel| format!(" q_gemv_vs_ours_rel={rel:.3e}"));
    let w_raw = q3k_gemv(cx, &cx.q3[&w_name], &x_w)?;
    if q_raw.len() != nq || w_raw.len() != HEADS {
        return Err(format!(
            "layer {l}: the projections are {} and {} wide",
            q_raw.len(),
            w_raw.len()
        )
        .into());
    }

    // The launch, and a rerun.
    let stream = cx.gpu.stream();
    let stride = k + LIST_SLACK;
    let nv = [u32::try_from(n_vis)?];
    let mut launch = Launch::new(stream, &q_raw, &w_raw, &nv, k, &cs, &keys, stride)?;
    let a = launch.run(&cx.kernels, stream)?;
    let rerun = same_out(&a, &launch.run(&cx.kernels, stream)?);

    // (i) The query and weights against our rules on the gemv's own output.
    let q_rule = bits_equal(&a.q, &query(&q_raw, &cs));
    let host_w: Vec<f32> = w_raw.iter().map(|&v| v * scale).collect();
    let w_rule = bits_equal(&a.w, &host_w);
    let i_ok = q_rule && q_gemv_ok && w_rule && tables_same;

    // (ii) The scores against our rule.
    let r = rule(&a.q[..nq], &a.w[..HEADS], &keys, n_vis);
    let got = &a.scores[..n_vis];
    let (mut over, mut worst) = (0usize, 0.0f64);
    for ((&x, &s), &bd) in got.iter().zip(&r.score).zip(&r.kernel) {
        let d = (f64::from(x) - s).abs();
        over += usize::from(outside(d, bd));
        worst = worst.max(d / bd);
    }
    let past = a.scores[n_vis..]
        .iter()
        .all(|v| v.to_bits() == POISON.to_bits());
    let ii_ok = over == 0 && past;

    // (iii) The ids: the rule's selection outside the tie band, the exact
    // top-k of the kernel's own scores, the list's shape.
    let (rule_off, band_rows) = ids_vs_rule(&a.list[..k], &r.score, &r.kernel, k);
    let exact = a.list[..k] == exact_top(got, k)[..];
    let shape = list_shape(&a.list, k, n_vis);
    let iii_ok = rule_off == 0 && exact && shape;

    // PIN(2026-11-18): the real-file ik set retired; ik cannot read bq3km's engram schema; no fixture set carries this clause
    let pass = i_ok && ii_ok && iii_ok && a.hist_zero && rerun;
    println!(
        "layer step={label} layer={l} n_vis={n_vis} rows={rows} top_k={k} — (i) q_rule_bits={q_rule} \
             w_rule_bits={w_rule}{q_gemv_line} table_is_ggml={tables_same} — (ii) rule_over={over}/{n_vis} \
             rule_worst={worst:.3} past_n_vis_untouched={past} — (iii) rule_off={rule_off} \
             tie_band_rows={band_rows} exact_top={exact} list_shape={shape} hist_zero={} \
             bit_identical_rerun={rerun} {}",
        a.hist_zero,
        verdict(pass)
    );
    Ok(pass)
}

// ------------------------------------------------------------ identity

/// (iv) Every emitting layer of the identity step: the list is `0..n_vis`,
/// and nothing else is written — the scores, query and weights stay poisoned
/// and the histogram zero.
fn identity_step(cx: &mut Cx, case: &StepCase) -> Result<bool, GateError> {
    let step = step_of(cx, case)?;
    let label = case.label;
    println!(
        "step {label}: position {}, top_k {}, context {}",
        step.pos, step.top_k, step.ctx
    );
    let k = step.top_k;
    let stride = k + LIST_SLACK;
    let (cs, _) = cx.tables(step.pos)?;
    let mut ok = true;
    for l in 0..cx.hp.layers.len() {
        if !cx.hp.layers[l].indexer {
            continue;
        }
        let (n_vis, rows) = layer_rows(&step, l)?;
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
            "identity step={label} layer={l} n_vis={n_vis} top_k={k} rows={rows} list_is_0..n_vis={identity} \
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
            let mut part_v =
                DeviceBuffer::from_host(stream, &vec![POISON; attn::partials_v_len(heads, segs)])?;
            let mut part_ms =
                DeviceBuffer::from_host(stream, &vec![POISON; attn::partials_ms_len(heads, segs)])?;
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
    let mut group: Vec<usize> = copies;
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
            want.list[..k] == exact_top(&want.scores[..n], k)[..] && list_shape(&want.list, k, n)
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
