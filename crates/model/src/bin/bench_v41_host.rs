//! Bench for the host leg of a DeepSeek-V4.1 decode token.
//!
//! The leg is the routed experts the CPU pool computes, at the file's own
//! shapes, read in place from the file's own page-cache pages. It is not a
//! gate; the verdicts it prints guard its own measurement.
//!
//! Per token, every layer runs the dispatch shape the host tier issues
//! (`moe::HostLayer`), over `n_host` routed experts: gate and up of every
//! expert in one `ops::matmul_q_group_into`, then every down projection in one
//! `ops::matmul_q_group_swiglu_into` whose inputs are the SwiGLU combines. The
//! shared expert is not part of the leg (q8_0, on a card in every placement
//! plan). Every matrix is cut from its stack's `ops::ShardTensor`, which reads
//! the shard that holds it, so a layer whose stacks sit in different shards
//! still runs one group of each; the start-up table names those layers. A
//! layer's output does not feed the next: in serving the card sits between
//! them. The per-matrix shape runs the same rows as one dispatch per expert
//! matrix.
//!
//! Weights: `gguf::Split` over `$BLOOMERY_REF_MODEL`, mapped lazily, never
//! copied — file-backed page-cache pages, as serving reads them. A layer's
//! routed stacks are its three-dimensional tensors whose last dim is the
//! file's `expert_count`: down by its per-expert shape `[ff, embd]`, gate and
//! up (both `[embd, ff]`) by the word their names carry. Each layer's working
//! set is [`WORKING_SET`] experts drawn once by a seeded generator, and a
//! token draws its `n_host` experts from that set, distinct within the
//! layer. The working set is paged in before anything runs
//! (`madvise(MADV_WILLNEED)`, then one read per page). Around every timed
//! arm its page-cache residency (`mincore`) and the process's page-fault
//! counters are recorded: a page read from NVMe makes the number a disk
//! measurement, and such an arm prints `admissible=no`.
//!
//! `--check`: for layers 0, 1, 2, the last layer and every layer whose
//! stacks span more than one shard, one token and every one of its
//! `expert_used_count` experts. Six output rows of gate, up and down are
//! compared with an f64 reference over the same bytes: `gguf::dequant_row`
//! for the weight rows, and for the activation the bytes `qdot::quantize_col`
//! makes of the kernel's own input (block_q8_K for q3_K, block_q8_2_x4 for
//! q4_K and q5_K), decoded exactly — the float sum order is the only
//! difference. The band is [`BAND`]. The SwiGLU combine the down dispatch
//! produced must equal `qdot::swiglu` over the gate and up outputs bit for
//! bit and sit within the band of an f64 SiLU, and the per-matrix shape must
//! reproduce the engine shape's outputs bit for bit. A miss names the layer,
//! the expert and the matrix, and the run exits non-zero.
//!
//! `--time` (lead-only, under `tools/ref/host-rate.sh`, which owns the lease,
//! the witnesses and the thread sweep): the check first, refusing to time if
//! it fails; then `--rounds` rounds of every arm, the arm order rotated each
//! round. Per arm and round, `--warmup` untimed tokens, then as many timed
//! tokens as fill the arm's share of `--seconds` at the first warm-up's pace;
//! later rounds keep that count. Every dispatch is timed on its own too, by
//! kind and weight bytes: bytes per dispatch is the variable this bench is
//! for.

use std::ffi::{c_int, c_void};
use std::process::ExitCode;
use std::time::Instant;

use gguf::{GgmlType, Gguf, Split, TensorInfo, dequant_row};
use model::ModelError;
use model::moe::expert_view;
use model::ops::{
    GroupInput, ShardTensor, Tensor2, Weight, matmul_q, matmul_q_group_into, matmul_q_group_swiglu,
    matmul_q_group_swiglu_into,
};

type BenchError = Box<dyn std::error::Error>;

/// Experts per layer in the working set: forty layers of them are far above
/// the L3 and far below free RAM.
const WORKING_SET: usize = 24;
/// Relative band of a checked value against its f64 reference, the GPU
/// bench's `KERNEL_BAND`: accumulation rounding sits orders of magnitude
/// under it, a wrong row, scale or activation orders of magnitude above.
const BAND: f64 = 1e-5;
/// Distinct activation columns; token `t` of layer `l` reads `(t + l) % N_X`.
const N_X: usize = 16;
/// Seed of every generator in the run; each use adds its own key.
const SEED: u64 = 0x9E37_79B9_7F4A_7C15;
const KEY_SET: u64 = 1;
const KEY_TOKEN: u64 = 2;
const KEY_CHECK: u64 = 3;
const KEY_X: u64 = 4;
/// The arms of a `--time` run without `--arms`: the engine shape with all six
/// experts on the host, five (about the host share of placement plan (a)) and
/// three (a hot-set placement), and the per-matrix shape at six.
const DEFAULT_ARMS: &str = "engine:6,engine:5,engine:3,per-matrix:6";
/// Bounds on the timed tokens of one arm and round.
const MIN_TOKENS: usize = 5;
const MAX_TOKENS: usize = 2000;
/// Stack order in every per-layer array: `[gate, up, down]`.
const GATE: usize = 0;
const UP: usize = 1;
const DOWN: usize = 2;
const MATRIX: [&str; 3] = ["gate", "up", "down"];
/// Pool dispatches of the engine shape per layer: the gate+up group and the
/// down group.
const ENGINE_DISPATCHES: usize = 2;

const USAGE: &str = "usage: bench_v41_host --check
       bench_v41_host --time [--rounds N] [--seconds S] [--warmup W] [--arms A,B,...]
  an arm is <engine|engine-sep|per-matrix>:<n_host>[x<rows>]; the default is engine:6,engine:5,engine:3,per-matrix:6
  the model is $BLOOMERY_REF_MODEL (tools/box.sh exports it from the deepseek41 profile)";

// The page bookkeeping's libc calls; `std` already links libc on this target.
unsafe extern "C" {
    fn mincore(addr: *mut c_void, length: usize, vec: *mut u8) -> c_int;
    fn madvise(addr: *mut c_void, length: usize, advice: c_int) -> c_int;
    safe fn getpagesize() -> c_int;
}

/// `MADV_WILLNEED` in the Linux UAPI (`asm-generic/mman-common.h`).
const MADV_WILLNEED: c_int = 3;

/// The hyperparameters the bench reads from the file, never from literals.
struct Meta {
    blocks: usize,
    embd: usize,
    ff: usize,
    n_expert: usize,
    n_used: usize,
}

impl Meta {
    fn read(split: &Split) -> Result<Meta, BenchError> {
        let get = |suffix: &str| -> Result<usize, BenchError> {
            let v = split
                .arch_get_u64(suffix)
                .ok_or_else(|| format!("metadata key {} is absent", split.arch_key(suffix)))?;
            Ok(usize::try_from(v)?)
        };
        let meta = Meta {
            blocks: get("block_count")?,
            embd: get("embedding_length")?,
            ff: get("expert_feed_forward_length")?,
            n_expert: get("expert_count")?,
            n_used: get("expert_used_count")?,
        };
        if meta.n_expert < WORKING_SET || meta.n_used == 0 || meta.n_used > WORKING_SET {
            return Err(format!(
                "{} experts, {} used per token: a working set of {WORKING_SET} does not fit",
                meta.n_expert, meta.n_used
            )
            .into());
        }
        Ok(meta)
    }
}

/// One routed stack as the file holds it: its shard and its header entry.
#[derive(Clone, Copy)]
struct Stack<'a> {
    shard: usize,
    info: &'a TensorInfo,
}

/// The layer number and the rest of a `blk.<layer>.<rest>` tensor name.
fn layer_part(name: &str) -> Option<(usize, &str)> {
    let (layer, rest) = name.strip_prefix("blk.")?.split_once('.')?;
    Some((layer.parse().ok()?, rest))
}

/// Which matrix a routed stack holds (an index into `[gate, up, down]`):
/// down by its per-expert shape `[ff, embd]`, gate or up — both
/// `[embd, ff]` — by the one of those two words its name carries. Anything
/// else is refused rather than guessed.
fn matrix_of(t: &TensorInfo, rest: &str, meta: &Meta) -> Result<usize, BenchError> {
    let (k, n) = (t.dims[0], t.dims[1]);
    let (embd, ff) = (meta.embd as u64, meta.ff as u64);
    if (k, n) == (ff, embd) {
        return Ok(DOWN);
    }
    if (k, n) != (embd, ff) {
        return Err(format!(
            "{}: {k} x {n} per expert is neither [embd, ff] nor [ff, embd]",
            t.name
        )
        .into());
    }
    let words: Vec<&str> = rest.split(['_', '.']).collect();
    match (words.contains(&"gate"), words.contains(&"up")) {
        (true, false) => Ok(GATE),
        (false, true) => Ok(UP),
        _ => Err(format!(
            "{}: an [embd, ff] stack whose name says neither gate nor up alone",
            t.name
        )
        .into()),
    }
}

/// Every layer's routed stacks, `[gate, up, down]`.
fn find_stacks<'a>(split: &'a Split, meta: &Meta) -> Result<Vec<[Stack<'a>; 3]>, BenchError> {
    let mut found: Vec<[Option<Stack<'a>>; 3]> = vec![[None; 3]; meta.blocks];
    for (shard, t) in split.iter_tensors() {
        if t.dims.len() != 3 || t.dims[2] != meta.n_expert as u64 {
            continue;
        }
        let (layer, rest) = layer_part(&t.name).ok_or_else(|| {
            format!(
                "{}: a stack of {} experts outside a layer",
                t.name, meta.n_expert
            )
        })?;
        let m = matrix_of(t, rest, meta)?;
        let slots = found.get_mut(layer).ok_or_else(|| {
            format!(
                "{}: layer {layer} is past the file's {} blocks",
                t.name, meta.blocks
            )
        })?;
        if slots[m].replace(Stack { shard, info: t }).is_some() {
            return Err(format!("layer {layer} holds two {} stacks", MATRIX[m]).into());
        }
    }
    found
        .into_iter()
        .enumerate()
        .map(|(l, s)| match s {
            [Some(g), Some(u), Some(d)] => Ok([g, u, d]),
            _ => Err(format!(
                "layer {l} lacks a routed stack (gate, up, down present: {}, {}, {})",
                s[GATE].is_some(),
                s[UP].is_some(),
                s[DOWN].is_some()
            )
            .into()),
        })
        .collect()
}

/// One expert of a layer's working set: its id, its `[gate, up, down]` views
/// cut by the engine's `moe::expert_view` (the per-matrix shape's), and the
/// same matrices as dispatch weights cut from the stacks' `ShardTensor`s (the
/// engine shape's).
struct Expert<'a> {
    id: usize,
    views: [TensorInfo; 3],
    weights: [Weight<'a>; 3],
}

/// One layer as the bench runs it: its stacks, the reader of each stack's
/// shard, and the working set.
struct Layer<'a> {
    index: usize,
    stacks: [Stack<'a>; 3],
    readers: [&'a Gguf; 3],
    experts: Vec<Expert<'a>>,
}

impl<'a> Layer<'a> {
    /// Matrix `m` of working-set slot `s`.
    fn view(&self, s: usize, m: usize) -> &TensorInfo {
        &self.experts[s].views[m]
    }

    /// Matrix `m` of working-set slot `s`, as a dispatch weight.
    fn weight(&self, s: usize, m: usize) -> Weight<'a> {
        self.experts[s].weights[m]
    }

    /// File bytes one expert of this layer reads: its three matrices.
    fn expert_bytes(&self) -> u64 {
        self.experts
            .first()
            .map_or(0, |e| e.views.iter().map(|v| v.nbytes).sum())
    }

    /// The three stacks span more than one shard.
    fn spans_shards(&self) -> bool {
        self.stacks[UP].shard != self.stacks[GATE].shard
            || self.stacks[DOWN].shard != self.stacks[GATE].shard
    }
}

/// Every layer with its readers and its working set's views, built once.
fn build_layers<'a>(split: &'a Split, meta: &Meta) -> Result<Vec<Layer<'a>>, BenchError> {
    let mut layers = Vec::with_capacity(meta.blocks);
    for (index, stacks) in find_stacks(split, meta)?.into_iter().enumerate() {
        let reader = |m: usize| -> Result<&'a Gguf, BenchError> {
            let s = stacks[m].shard;
            split
                .shard(s)
                .ok_or_else(|| format!("layer {index}: the split has no shard {s}").into())
        };
        let readers = [reader(GATE)?, reader(UP)?, reader(DOWN)?];
        let handle = |m: usize| ShardTensor::find(split, &stacks[m].info.name);
        let handles = [handle(GATE)?, handle(UP)?, handle(DOWN)?];
        let ids = distinct(
            &mut Rng::new(&[KEY_SET, index as u64]),
            WORKING_SET,
            meta.n_expert,
        );
        let mut experts = Vec::with_capacity(ids.len());
        for id in ids {
            let view = |m: usize| expert_view(stacks[m].info, id, meta.n_expert);
            let weight = |m: usize| handles[m].expert(split, id);
            experts.push(Expert {
                id,
                views: [view(GATE)?, view(UP)?, view(DOWN)?],
                weights: [weight(GATE)?, weight(UP)?, weight(DOWN)?],
            });
        }
        layers.push(Layer {
            index,
            stacks,
            readers,
            experts,
        });
    }
    Ok(layers)
}

/// xorshift64*, seeded from a key so every draw replays from its key alone.
struct Rng(u64);

impl Rng {
    fn new(key: &[u64]) -> Rng {
        Rng(key.iter().fold(SEED, |s, &k| splitmix(s ^ k)) | 1)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// A value below `bound`.
    fn below(&mut self, bound: usize) -> usize {
        (self.next_u64() % bound as u64) as usize
    }
}

/// splitmix64's finalizer: one well-mixed word per key step.
fn splitmix(z: u64) -> u64 {
    let z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    let z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// `n` distinct values below `bound`, in draw order.
fn distinct(rng: &mut Rng, n: usize, bound: usize) -> Vec<usize> {
    assert!(n <= bound, "{n} distinct values below {bound}");
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let v = rng.below(bound);
        if !out.contains(&v) {
            out.push(v);
        }
    }
    out
}

/// [`N_X`] seeded activation columns of `embd` values in [-1, 1), every 61st
/// scaled by 8, so a block's scale is set by an outlier the way real
/// activations set it.
fn activations(embd: usize) -> Vec<Tensor2> {
    (0..N_X)
        .map(|c| {
            let mut rng = Rng::new(&[KEY_X, c as u64]);
            let data = (0..embd)
                .map(|i| {
                    let u = (rng.next_u64() >> 40) as f32 / 8_388_608.0 - 1.0;
                    if i.is_multiple_of(61) { u * 8.0 } else { u }
                })
                .collect();
            Tensor2::from_vec(embd, 1, data)
        })
        .collect()
}

/// What one layer's dispatches produced, in the order of the token's experts.
struct LayerOut {
    gate: Vec<Tensor2>,
    up: Vec<Tensor2>,
    down: Vec<Tensor2>,
    par: Vec<Tensor2>,
}

/// Wall time of every dispatch, by what it read: a kind and its weight
/// bytes. A handful of entries, found by a scan.
#[derive(Default)]
struct Tally {
    rows: Vec<TallyRow>,
}

struct TallyRow {
    kind: &'static str,
    bytes: u64,
    count: u64,
    ns: u64,
}

impl Tally {
    fn add(&mut self, kind: &'static str, bytes: u64, t0: Instant) {
        let ns = t0.elapsed().as_nanos() as u64;
        match self
            .rows
            .iter_mut()
            .find(|r| r.kind == kind && r.bytes == bytes)
        {
            Some(r) => {
                r.count += 1;
                r.ns += ns;
            }
            None => self.rows.push(TallyRow {
                kind,
                bytes,
                count: 1,
                ns,
            }),
        }
    }
}

/// Weight bytes a dispatch reads.
fn bytes_of(ws: &[Weight<'_>]) -> u64 {
    ws.iter().map(|w| w.bytes().len() as u64).sum()
}

/// One output block per weight, off the pool, for an `_into` dispatch.
fn blocks(ws: &[Weight<'_>]) -> Vec<Tensor2> {
    ws.iter().map(|w| Tensor2::scratch(w.n(), 1)).collect()
}

/// The engine's shape: gate and up of every expert in one group, the
/// weights interleaved per expert as the host tier lays them out, each read
/// from its own shard, then one down group whose inputs are the SwiGLU
/// combines.
fn engine_layer(
    l: &Layer<'_>,
    slots: &[usize],
    x: &Tensor2,
    tally: &mut Tally,
) -> Result<LayerOut, ModelError> {
    let n = slots.len();
    let xs = vec![x; 2 * n];
    let ws: Vec<Weight<'_>> = slots
        .iter()
        .flat_map(|&s| [l.weight(s, GATE), l.weight(s, UP)])
        .collect();
    let mut gu = blocks(&ws);
    let t0 = Instant::now();
    matmul_q_group_into("host_gate_up", &ws, &xs, &mut gu)?;
    tally.add("gate+up", bytes_of(&ws), t0);
    let (mut gate, mut up) = (Vec::with_capacity(n), Vec::with_capacity(n));
    let mut gu = gu.into_iter();
    while let (Some(g), Some(u)) = (gu.next(), gu.next()) {
        gate.push(g);
        up.push(u);
    }
    let dw: Vec<Weight<'_>> = slots.iter().map(|&s| l.weight(s, DOWN)).collect();
    let srcs: Vec<GroupInput<'_>> = gate
        .iter()
        .zip(&up)
        .map(|(g, u)| GroupInput::Swiglu(g, u))
        .collect();
    let (mut down, mut par) = (
        blocks(&dw),
        gate.iter()
            .map(|g| Tensor2::scratch(g.ne0, 1))
            .collect::<Vec<_>>(),
    );
    let t0 = Instant::now();
    matmul_q_group_swiglu_into("host_down", &dw, &srcs, &mut down, &mut par)?;
    tally.add("down", bytes_of(&dw), t0);
    drop(srcs);
    Ok(LayerOut {
        gate,
        up,
        down,
        par,
    })
}

/// Probe shape for a k-token step: `xs.len()` rows, row `i` owning the
/// `slots.len() / xs.len()` slots `slots[i * n..(i + 1) * n]` and reading
/// `xs[i]`. Every row's gate and up in one group, every row's down in one
/// group — the dispatches a tier that takes k tokens per call would issue.
fn engine_rows_layer(
    l: &Layer<'_>,
    slots: &[usize],
    xs: &[&Tensor2],
    tally: &mut Tally,
) -> Result<(), ModelError> {
    let per_row = slots.len() / xs.len();
    let xin: Vec<&Tensor2> = (0..slots.len())
        .flat_map(|j| [xs[j / per_row]; 2])
        .collect();
    let ws: Vec<Weight<'_>> = slots
        .iter()
        .flat_map(|&s| [l.weight(s, GATE), l.weight(s, UP)])
        .collect();
    let mut gu = blocks(&ws);
    let t0 = Instant::now();
    matmul_q_group_into("host_gate_up", &ws, &xin, &mut gu)?;
    tally.add("gate+up", bytes_of(&ws), t0);
    let dw: Vec<Weight<'_>> = slots.iter().map(|&s| l.weight(s, DOWN)).collect();
    let (pairs, _) = gu.as_chunks::<2>();
    let srcs: Vec<GroupInput<'_>> = pairs
        .iter()
        .map(|[g, u]| GroupInput::Swiglu(g, u))
        .collect();
    let mut down = blocks(&dw);
    let mut par: Vec<Tensor2> = pairs
        .iter()
        .map(|[g, _]| Tensor2::scratch(g.ne0, 1))
        .collect();
    let t0 = Instant::now();
    matmul_q_group_swiglu_into("host_down", &dw, &srcs, &mut down, &mut par)?;
    tally.add("down", bytes_of(&dw), t0);
    Ok(())
}

/// One dispatch per expert matrix: gate and up through `ops::matmul_q`, down
/// through a one-pair `matmul_q_group_swiglu` over that expert's combine.
fn per_matrix_layer(
    l: &Layer<'_>,
    slots: &[usize],
    x: &Tensor2,
    tally: &mut Tally,
) -> Result<LayerOut, ModelError> {
    let n = slots.len();
    let mut out = LayerOut {
        gate: Vec::with_capacity(n),
        up: Vec::with_capacity(n),
        down: Vec::with_capacity(n),
        par: Vec::with_capacity(n),
    };
    for &s in slots {
        let (gw, uw, dw) = (l.view(s, GATE), l.view(s, UP), l.view(s, DOWN));
        let t0 = Instant::now();
        let g = matmul_q(l.readers[GATE], gw, x)?;
        tally.add("gate", gw.nbytes, t0);
        let t0 = Instant::now();
        let u = matmul_q(l.readers[UP], uw, x)?;
        tally.add("up", uw.nbytes, t0);
        let t0 = Instant::now();
        let (down, par) =
            matmul_q_group_swiglu(l.readers[DOWN], &[dw], &[GroupInput::Swiglu(&g, &u)])?;
        tally.add("down", dw.nbytes, t0);
        out.down.extend(down);
        out.par.extend(par);
        out.gate.push(g);
        out.up.push(u);
    }
    Ok(out)
}

/// The working set in the mappings: one byte span per expert matrix. A page
/// two neighbouring experts share counts once in each span.
struct Pages<'a> {
    spans: Vec<&'a [u8]>,
    page: usize,
}

impl<'a> Pages<'a> {
    fn of(layers: &[Layer<'a>]) -> Result<Pages<'a>, BenchError> {
        let page = usize::try_from(getpagesize())?;
        let mut spans = Vec::new();
        for l in layers {
            for e in &l.experts {
                for (m, v) in e.views.iter().enumerate() {
                    spans.push(l.readers[m].data(v)?);
                }
            }
        }
        Ok(Pages { spans, page })
    }

    /// `s` widened to whole pages: its first page's address and the length
    /// through its last page.
    fn aligned(&self, s: &[u8]) -> (*mut c_void, usize) {
        let lead = s.as_ptr().addr() % self.page;
        let len = (lead + s.len()).div_ceil(self.page) * self.page;
        (s.as_ptr().wrapping_sub(lead).cast_mut().cast(), len)
    }

    /// Pages the spans cover, summed span by span.
    fn total(&self) -> usize {
        self.spans
            .iter()
            .map(|s| self.aligned(s).1 / self.page)
            .sum()
    }

    /// Weight bytes the spans hold.
    fn bytes(&self) -> u64 {
        self.spans.iter().map(|s| s.len() as u64).sum()
    }

    /// Pages of the spans the page cache holds now (`mincore`, bit 0).
    fn resident(&self) -> Result<usize, BenchError> {
        let mut vec: Vec<u8> = Vec::new();
        let mut n = 0;
        for s in &self.spans {
            let (addr, len) = self.aligned(s);
            vec.resize(len / self.page, 0);
            // SAFETY: `addr..addr + len` is `s` widened to whole pages, and a
            // file mapping covers whole pages, so the range lies inside the
            // shard's live mapping. mincore only reads the range's page state
            // and writes one byte per page into `vec`, which holds that many.
            let rc = unsafe { mincore(addr, len, vec.as_mut_ptr()) };
            if rc != 0 {
                return Err(format!("mincore: {}", std::io::Error::last_os_error()).into());
            }
            n += vec.iter().filter(|&&b| b & 1 == 1).count();
        }
        Ok(n)
    }

    /// Page the working set in: WILLNEED readahead over every span, then one
    /// read in every page, so no timed token takes a fault of its own.
    fn populate(&self) -> Result<(), BenchError> {
        for s in &self.spans {
            let (addr, len) = self.aligned(s);
            // SAFETY: the range `resident` reads, inside the live mapping.
            // WILLNEED only starts readahead of the file pages behind it; no
            // byte of the mapping changes.
            let rc = unsafe { madvise(addr, len, MADV_WILLNEED) };
            if rc != 0 {
                return Err(format!("madvise: {}", std::io::Error::last_os_error()).into());
            }
        }
        let mut acc = 0u64;
        for s in &self.spans {
            // The first byte, then the first byte of every later page.
            let mut i = 0;
            while i < s.len() {
                acc = acc.wrapping_add(u64::from(s[i]));
                i += self.page - (s.as_ptr().addr() + i) % self.page;
            }
        }
        std::hint::black_box(acc);
        Ok(())
    }
}

/// The process's minor and major page faults so far, every thread counted
/// (`/proc/self/stat` fields 10 and 12).
fn faults() -> Result<(u64, u64), BenchError> {
    let stat = std::fs::read_to_string("/proc/self/stat")?;
    let after = stat
        .rsplit_once(')')
        .ok_or("/proc/self/stat has no comm field")?
        .1;
    let f: Vec<&str> = after.split_whitespace().collect();
    // `after` opens at field 3.
    let field = |k: usize| -> Result<u64, BenchError> {
        Ok(f.get(k - 3).ok_or("/proc/self/stat is short")?.parse()?)
    };
    Ok((field(10)?, field(12)?))
}

/// One line of `/sys/devices/system/cpu/cpu<cpu>/<what>`.
fn cpu_sys(cpu: usize, what: &str) -> Result<String, BenchError> {
    let path = format!("/sys/devices/system/cpu/cpu{cpu}/{what}");
    Ok(std::fs::read_to_string(path)?.trim().to_string())
}

/// Every thread of the process, read back from the kernel: the cpus it may
/// run on and, when that is one cpu, its core, the core's SMT siblings and
/// its L3 group. This is what the pool's pinning did, not what it meant to.
fn print_affinity() -> Result<(), BenchError> {
    let mut rows: Vec<(u64, String, String)> = Vec::new();
    for entry in std::fs::read_dir("/proc/self/task")? {
        let dir = entry?.path();
        let tid = dir
            .file_name()
            .and_then(|n| n.to_str())
            .and_then(|n| n.parse().ok())
            .ok_or("a /proc/self/task entry that is not a thread id")?;
        let comm = std::fs::read_to_string(dir.join("comm"))?
            .trim()
            .to_string();
        let status = std::fs::read_to_string(dir.join("status"))?;
        let cpus = status
            .lines()
            .find_map(|l| l.strip_prefix("Cpus_allowed_list:"))
            .ok_or("a task status without Cpus_allowed_list")?
            .trim()
            .to_string();
        rows.push((tid, comm, cpus));
    }
    // Thread ids in spawn order: the main thread, then the workers in pool order
    // (their names are cut to 15 bytes, so the name alone cannot order them).
    rows.sort_unstable_by_key(|r| r.0);
    let (mut cores, mut l3s): (Vec<String>, Vec<(String, usize)>) = (Vec::new(), Vec::new());
    for (tid, comm, cpus) in &rows {
        let Ok(cpu) = cpus.parse::<usize>() else {
            println!("v41host thread tid={tid} name={comm} cpus={cpus} pinned=no");
            continue;
        };
        let core = cpu_sys(cpu, "topology/core_id")?;
        let siblings = cpu_sys(cpu, "topology/thread_siblings_list")?;
        let l3 = cpu_sys(cpu, "cache/index3/shared_cpu_list")?;
        println!(
            "v41host thread tid={tid} name={comm} cpu={cpu} core={core} smt_siblings={siblings} l3={l3}"
        );
        if !cores.contains(&core) {
            cores.push(core);
        }
        match l3s.iter_mut().find(|(k, _)| *k == l3) {
            Some((_, n)) => *n += 1,
            None => l3s.push((l3, 1)),
        }
    }
    let pinned: usize = l3s.iter().map(|(_, n)| n).sum();
    let per_l3: Vec<String> = l3s.iter().map(|(k, n)| format!("{k}:{n}")).collect();
    println!(
        "v41host pinning threads={} pinned={pinned} distinct_cores={} per_l3=[{}]",
        rows.len(),
        cores.len(),
        per_l3.join(" ")
    );
    Ok(())
}

/// The resident set as the kernel accounts it, and how much of the
/// file-backed part sits in PMD-sized mappings.
fn print_mem() -> Result<(), BenchError> {
    let s = std::fs::read_to_string("/proc/self/smaps_rollup")?;
    let keep: Vec<String> = s
        .lines()
        .filter(|l| {
            ["Rss:", "FilePmdMapped:", "AnonHugePages:"]
                .iter()
                .any(|k| l.starts_with(k))
        })
        .map(|l| l.split_whitespace().collect::<Vec<_>>().join(""))
        .collect();
    println!("v41host mem {}", keep.join(" "));
    Ok(())
}

/// The start-up table: the file's shape, each layer's shards and types, and
/// the layers whose stacks span shards.
fn print_table(path: &str, split: &Split, meta: &Meta, layers: &[Layer<'_>]) {
    println!(
        "v41host model={path} shards={} blocks={} embd={} ff={} experts={} used={} working_set={WORKING_SET}",
        split.shard_count(),
        meta.blocks,
        meta.embd,
        meta.ff,
        meta.n_expert,
        meta.n_used
    );
    for l in layers {
        let st = &l.stacks;
        println!(
            "v41host layer={} gate={}@shard{} up={}@shard{} down={}@shard{} expert_bytes={} engine_dispatches={}",
            l.index,
            st[GATE].info.ty,
            st[GATE].shard,
            st[UP].info.ty,
            st[UP].shard,
            st[DOWN].info.ty,
            st[DOWN].shard,
            l.expert_bytes(),
            ENGINE_DISPATCHES
        );
    }
    let spanning: Vec<usize> = layers
        .iter()
        .filter(|l| l.spans_shards())
        .map(|l| l.index)
        .collect();
    println!(
        "v41host shard_spanning_layers={spanning:?} (one gate+up group and one down group each, every matrix read from its own shard)"
    );
}

/// The output rows a check compares: both ends and rows a third, a half and
/// two thirds in.
fn check_rows(n: usize) -> Vec<usize> {
    let mut v = vec![0, 1, n / 3, n / 2, 2 * n / 3 + 1, n - 1];
    v.retain(|&o| o < n);
    v.sort_unstable();
    v.dedup();
    v
}

/// `max |got - want| / max |want|`; a non-finite value or an all-zero
/// reference is an error, not a pass.
fn rel_err(got: &[f32], want: &[f64]) -> Result<f64, String> {
    if got.len() != want.len() {
        return Err(format!("{} values against {}", got.len(), want.len()));
    }
    if let Some(i) = got.iter().position(|v| !v.is_finite()) {
        return Err(format!("non-finite output at {i}"));
    }
    if let Some(i) = want.iter().position(|v| !v.is_finite()) {
        return Err(format!("non-finite reference at {i}"));
    }
    let denom = want.iter().fold(0.0f64, |a, &w| a.max(w.abs()));
    if denom == 0.0 {
        return Err("the reference is all zero".to_string());
    }
    let num = got
        .iter()
        .zip(want)
        .fold(0.0f64, |a, (&g, &w)| a.max((f64::from(g) - w).abs()));
    Ok(num / denom)
}

/// Same length and the same bits everywhere.
fn bits_equal(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// The activation values a fused kernel for weight type `ty` reads: `x`
/// through `qdot::quantize_col`, the bytes decoded exactly in f64.
fn kernel_activation(ty: GgmlType, x: &[f32]) -> Result<Vec<f64>, BenchError> {
    let mut col = vec![0u8; qdot::col_bytes(ty, x.len())];
    qdot::quantize_col(ty, x, &mut col);
    let values = match ty {
        GgmlType::Q3_K => decode_q8k(&col),
        GgmlType::Q4_K | GgmlType::Q5_K => decode_q82x4(&col),
        other => return Err(format!("no activation decoder for {other}").into()),
    };
    if values.len() != x.len() {
        return Err(format!("{ty}: decoded {} values of {}", values.len(), x.len()).into());
    }
    Ok(values)
}

/// block_q8_K as `qdot::quantize_col` writes it: 296 bytes per 256 values —
/// the f32 scale at 0, the int8 codes at 8, the block sums after them.
fn decode_q8k(col: &[u8]) -> Vec<f64> {
    let (blocks, _) = col.as_chunks::<296>();
    blocks
        .iter()
        .flat_map(|b| {
            let d = f64::from(f32::from_le_bytes([b[0], b[1], b[2], b[3]]));
            b[8..264]
                .iter()
                .map(move |&q| d * f64::from(i8::from_le_bytes([q])))
        })
        .collect()
}

/// block_q8_2_x4 as `qdot::quantize_col` writes it: 144 bytes per 128 values
/// — four bf16 scales at 0, four block sums at 8, four runs of 32 int8 codes
/// from 16.
fn decode_q82x4(col: &[u8]) -> Vec<f64> {
    let (groups, _) = col.as_chunks::<144>();
    groups
        .iter()
        .flat_map(|g| {
            (0..4).flat_map(move |ir| {
                let bits = u16::from_le_bytes([g[2 * ir], g[2 * ir + 1]]);
                let d = f64::from(f32::from_bits(u32::from(bits) << 16));
                g[16 + 32 * ir..16 + 32 * (ir + 1)]
                    .iter()
                    .map(move |&q| d * f64::from(i8::from_le_bytes([q])))
            })
        })
        .collect()
}

/// f64 dot products of rows `rows` of `w` (read from its shard `g`) with `act`.
fn reference(
    g: &Gguf,
    w: &TensorInfo,
    act: &[f64],
    rows: &[usize],
) -> Result<Vec<f64>, BenchError> {
    let k = usize::try_from(w.dims[0])?;
    let n = usize::try_from(w.dims[1])?;
    if act.len() != k {
        return Err(format!("{}: {} activation values for k = {k}", w.name, act.len()).into());
    }
    let bytes = g.data(w)?;
    let rb = bytes.len() / n;
    let mut vals = vec![0.0f32; k];
    rows.iter()
        .map(|&r| -> Result<f64, BenchError> {
            dequant_row(w.ty, &bytes[r * rb..(r + 1) * rb], &mut vals)?;
            Ok(vals.iter().zip(act).map(|(&v, &a)| f64::from(v) * a).sum())
        })
        .collect()
}

/// The check's running state: whether every line prints, the sites that
/// failed, and the worst error compared with the band.
struct Checks {
    verbose: bool,
    sites: usize,
    failed: Vec<String>,
    worst: f64,
}

impl Checks {
    /// One site's verdict: its line prints when verbose or failing.
    fn record(&mut self, site: String, line: &str, pass: bool) {
        self.sites += 1;
        if self.verbose || !pass {
            println!("{line} {}", if pass { "PASS" } else { "FAIL" });
        }
        if !pass {
            self.failed.push(site);
        }
    }

    /// An error measure against the band: its printed form and whether it passed.
    fn measure(&mut self, err: Result<f64, String>) -> (String, bool) {
        match err {
            Ok(e) => {
                self.worst = self.worst.max(e);
                (format!("{e:.3e}"), e <= BAND)
            }
            Err(e) => (format!("[{e}]"), false),
        }
    }
}

/// Six rows of `got`, the kernel's output for matrix `m` of working-set slot
/// `s`, against the f64 reference over the same weight bytes and `act`.
fn check_weight(
    c: &mut Checks,
    l: &Layer<'_>,
    s: usize,
    m: usize,
    got: &Tensor2,
    act: &[f64],
) -> Result<(), BenchError> {
    let w = l.view(s, m);
    let rows = check_rows(got.data.len());
    let want = reference(l.readers[m], w, act, &rows)?;
    let got_rows: Vec<f32> = rows.iter().map(|&r| got.data[r]).collect();
    let (err, pass) = c.measure(rel_err(&got_rows, &want));
    let line = format!(
        "check layer={} expert={} matrix={} type={} k={} rows={} rows_checked={} max_rel_err={err} band={BAND:e}",
        l.index,
        l.experts[s].id,
        MATRIX[m],
        w.ty,
        w.dims[0],
        w.dims[1],
        rows.len()
    );
    let site = format!("layer {} expert {} {}", l.index, l.experts[s].id, MATRIX[m]);
    c.record(site, &line, pass);
    Ok(())
}

/// The combine the down dispatch produced for slot `s`, every value: bit for
/// bit `qdot::swiglu` over the same gate and up outputs, and within the band
/// of an f64 SiLU.
fn check_swiglu(
    c: &mut Checks,
    l: &Layer<'_>,
    s: usize,
    gate: &Tensor2,
    up: &Tensor2,
    par: &Tensor2,
) {
    let mut engine = vec![0.0f32; gate.data.len()];
    qdot::swiglu(&gate.data, &up.data, &mut engine);
    let same = bits_equal(&par.data, &engine);
    let want: Vec<f64> = gate
        .data
        .iter()
        .zip(&up.data)
        .map(|(&g, &u)| {
            let g = f64::from(g);
            g / (1.0 + (-g).exp()) * f64::from(u)
        })
        .collect();
    let (err, close) = c.measure(rel_err(&par.data, &want));
    let line = format!(
        "check layer={} expert={} matrix=swiglu n={} bits_equal_qdot_swiglu={same} max_rel_err_f64={err} band={BAND:e}",
        l.index,
        l.experts[s].id,
        par.data.len()
    );
    let site = format!("layer {} expert {} swiglu", l.index, l.experts[s].id);
    c.record(site, &line, same && close);
}

/// The check of one layer for one token: both shapes' outputs bit-equal,
/// then every expert's gate, up, combine and down.
fn check_layer(
    c: &mut Checks,
    l: &Layer<'_>,
    slots: &[usize],
    x: &Tensor2,
) -> Result<(), BenchError> {
    let mut tally = Tally::default();
    let out = engine_layer(l, slots, x, &mut tally)?;
    let pm = per_matrix_layer(l, slots, x, &mut tally)?;
    let pairs = [
        (&out.gate, &pm.gate),
        (&out.up, &pm.up),
        (&out.down, &pm.down),
        (&out.par, &pm.par),
    ];
    let same = pairs.iter().all(|(a, b)| {
        a.len() == b.len()
            && a.iter()
                .zip(b.iter())
                .all(|(a, b)| bits_equal(&a.data, &b.data))
    });
    let line = format!(
        "check layer={} shape=per-matrix experts={} outputs_bits_equal_engine={same}",
        l.index,
        slots.len()
    );
    c.record(format!("layer {} per-matrix", l.index), &line, same);
    let act_gate = kernel_activation(l.view(slots[0], GATE).ty, &x.data)?;
    let act_up = kernel_activation(l.view(slots[0], UP).ty, &x.data)?;
    for (i, &s) in slots.iter().enumerate() {
        check_weight(c, l, s, GATE, &out.gate[i], &act_gate)?;
        check_weight(c, l, s, UP, &out.up[i], &act_up)?;
        check_swiglu(c, l, s, &out.gate[i], &out.up[i], &out.par[i]);
        let act_down = kernel_activation(l.view(s, DOWN).ty, &out.par[i].data)?;
        check_weight(c, l, s, DOWN, &out.down[i], &act_down)?;
    }
    Ok(())
}

/// The whole check (see the module doc); `Ok` only when every site passed.
fn check(
    layers: &[Layer<'_>],
    n_host: usize,
    xs: &[Tensor2],
    verbose: bool,
) -> Result<(), BenchError> {
    let last = layers.len().saturating_sub(1);
    let covered: Vec<usize> = layers
        .iter()
        .filter(|l| l.index < 3 || l.index == last || l.spans_shards())
        .map(|l| l.index)
        .collect();
    let mut c = Checks {
        verbose,
        sites: 0,
        failed: Vec::new(),
        worst: 0.0,
    };
    for &li in &covered {
        let slots = distinct(&mut Rng::new(&[KEY_CHECK, li as u64]), n_host, WORKING_SET);
        check_layer(&mut c, &layers[li], &slots, &xs[li % N_X])?;
    }
    println!(
        "check layers={covered:?} experts_per_layer={n_host} sites={} failed={} worst_rel_err={:.3e} band={BAND:e}",
        c.sites,
        c.failed.len(),
        c.worst
    );
    if !c.failed.is_empty() {
        return Err(format!("check failed for: {}", c.failed.join(", ")).into());
    }
    println!("PASSED: bench_v41_host check — every site within {BAND:e} of its f64 reference");
    Ok(())
}

/// A timed arm: the dispatch shape and the host's share of a token's experts.
#[derive(Clone, Copy)]
struct Arm {
    per_matrix: bool,
    n_host: usize,
    /// Tokens per step (probe): each row reads its own activation column
    /// and `n_host` experts no other row of the layer reads.
    rows: usize,
    /// With `rows > 1`: one engine-shape call per row instead of one group
    /// over every row.
    separate: bool,
}

impl Arm {
    fn parse(s: &str) -> Result<Arm, String> {
        let (shape, n) = s.split_once(':').ok_or_else(|| {
            format!("arm {s:?}: want <engine|engine-sep|per-matrix>:<n_host>[x<rows>]")
        })?;
        let (per_matrix, separate) = match shape {
            "engine" => (false, false),
            "engine-sep" => (false, true),
            "per-matrix" => (true, false),
            other => return Err(format!("arm {s:?}: unknown shape {other:?}")),
        };
        let (n, rows) = match n.split_once('x') {
            Some((n, r)) => (
                n,
                r.parse::<usize>()
                    .map_err(|_| format!("arm {s:?}: rows {r:?} is not a count"))?,
            ),
            None => (n, 1),
        };
        let n_host: usize = n
            .parse()
            .map_err(|_| format!("arm {s:?}: n_host {n:?} is not a count"))?;
        if n_host == 0 || rows == 0 || n_host * rows > WORKING_SET {
            return Err(format!(
                "arm {s:?}: n_host and rows must be positive, n_host x rows at most {WORKING_SET}"
            ));
        }
        if per_matrix && rows > 1 {
            return Err(format!("arm {s:?}: the per-matrix shape runs one row"));
        }
        Ok(Arm {
            per_matrix,
            n_host,
            rows,
            separate,
        })
    }

    fn label(self) -> String {
        let shape = match (self.per_matrix, self.separate) {
            (true, _) => "per-matrix",
            (false, true) => "engine-sep",
            (false, false) => "engine",
        };
        if self.rows == 1 {
            format!("{shape}:{}", self.n_host)
        } else {
            format!("{shape}:{}x{}", self.n_host, self.rows)
        }
    }

    /// One layer: `slots` holds `rows` runs of `n_host`, row `i` reading
    /// `xs[i]`.
    fn run_layer(
        self,
        l: &Layer<'_>,
        slots: &[usize],
        xs: &[&Tensor2],
        tally: &mut Tally,
    ) -> Result<(), ModelError> {
        if self.per_matrix {
            per_matrix_layer(l, slots, xs[0], tally).map(drop)
        } else if self.rows == 1 {
            engine_layer(l, slots, xs[0], tally).map(drop)
        } else if self.separate {
            for (row, x) in slots.chunks_exact(self.n_host).zip(xs) {
                engine_layer(l, row, x, tally)?;
            }
            Ok(())
        } else {
            engine_rows_layer(l, slots, xs, tally)
        }
    }

    /// File bytes one token of this arm reads.
    fn bytes_per_token(self, layers: &[Layer<'_>]) -> u64 {
        layers
            .iter()
            .map(|l| (self.n_host * self.rows) as u64 * l.expert_bytes())
            .sum()
    }

    /// Pool dispatches one token of this arm issues, by construction.
    fn dispatches_per_token(self, layers: &[Layer<'_>]) -> usize {
        if self.per_matrix {
            3 * self.n_host * layers.len()
        } else if self.separate {
            ENGINE_DISPATCHES * self.rows * layers.len()
        } else {
            ENGINE_DISPATCHES * layers.len()
        }
    }
}

/// What `--time` takes on the command line.
struct TimeOpts {
    rounds: usize,
    seconds: f64,
    warmup: usize,
    arms: Vec<Arm>,
}

enum Mode {
    Check,
    Time(TimeOpts),
}

fn parse_args(args: &[String]) -> Result<Mode, String> {
    let mut it = args.iter().map(String::as_str);
    match it.next() {
        Some("--check") => {
            if let Some(extra) = it.next() {
                return Err(format!("--check takes no options, got {extra:?}"));
            }
            Ok(Mode::Check)
        }
        Some("--time") => {
            let mut opts = TimeOpts {
                rounds: 3,
                seconds: 2.5,
                warmup: 3,
                arms: Vec::new(),
            };
            let mut arms = DEFAULT_ARMS.to_string();
            while let Some(flag) = it.next() {
                let value = it.next().ok_or_else(|| format!("{flag} needs a value"))?;
                let bad = || format!("{flag} {value:?} is not a valid value");
                match flag {
                    "--rounds" => opts.rounds = value.parse().map_err(|_| bad())?,
                    "--seconds" => opts.seconds = value.parse().map_err(|_| bad())?,
                    "--warmup" => opts.warmup = value.parse().map_err(|_| bad())?,
                    "--arms" => arms = value.to_string(),
                    other => return Err(format!("unknown option {other:?}")),
                }
            }
            let seconds_ok = opts.seconds.is_finite() && opts.seconds > 0.0;
            if opts.rounds == 0 || opts.warmup == 0 || !seconds_ok {
                return Err("--rounds, --warmup and --seconds must be positive".to_string());
            }
            opts.arms = arms.split(',').map(Arm::parse).collect::<Result<_, _>>()?;
            Ok(Mode::Time(opts))
        }
        _ => Err("want --check or --time".to_string()),
    }
}

/// What every timed round reads: the layers, the activation columns and the
/// working set's pages.
struct Bench<'a> {
    layers: Vec<Layer<'a>>,
    xs: Vec<Tensor2>,
    pages: Pages<'a>,
}

/// One arm's round: the timed tokens and the page state around them.
struct ArmRound {
    ms: Vec<f64>,
    resident_before: usize,
    resident_after: usize,
    minflt: u64,
    majflt: u64,
    dispatches: u64,
    tally: Tally,
}

impl Bench<'_> {
    /// The working-set slots token `t` of `round` reads, per layer.
    fn draw(&self, n_host: usize, round: usize, t: usize) -> Vec<Vec<usize>> {
        (0..self.layers.len())
            .map(|l| {
                let key = [KEY_TOKEN, round as u64, t as u64, l as u64];
                distinct(&mut Rng::new(&key), n_host, WORKING_SET)
            })
            .collect()
    }

    /// Every layer of one token.
    fn token(
        &self,
        arm: Arm,
        ids: &[Vec<usize>],
        t: usize,
        tally: &mut Tally,
    ) -> Result<(), ModelError> {
        for (l, layer) in self.layers.iter().enumerate() {
            let xs: Vec<&Tensor2> = (0..arm.rows).map(|i| &self.xs[(t + l + i) % N_X]).collect();
            arm.run_layer(layer, &ids[l], &xs, tally)?;
        }
        Ok(())
    }

    /// One round of `arm`: `warmup` untimed tokens, then `tokens` timed ones —
    /// or, when `None`, as many as fill `target_ms` at the warm-up's pace
    /// (its first token left out when there are more).
    fn round(
        &self,
        arm: Arm,
        round: usize,
        warmup: usize,
        tokens: Option<usize>,
        target_ms: f64,
    ) -> Result<ArmRound, BenchError> {
        let mut scratch = Tally::default();
        let mut warm_ms = Vec::with_capacity(warmup);
        for t in 0..warmup {
            let ids = self.draw(arm.n_host * arm.rows, round, t);
            let t0 = Instant::now();
            self.token(arm, &ids, t, &mut scratch)?;
            warm_ms.push(t0.elapsed().as_secs_f64() * 1e3);
        }
        let paced = &warm_ms[usize::from(warm_ms.len() > 1)..];
        let pace = paced.iter().sum::<f64>() / paced.len() as f64;
        let tokens = tokens
            .unwrap_or_else(|| ((target_ms / pace).ceil() as usize).clamp(MIN_TOKENS, MAX_TOKENS));
        let mut tally = Tally::default();
        let mut ms = Vec::with_capacity(tokens);
        let resident_before = self.pages.resident()?;
        let (min0, maj0) = faults()?;
        let d0 = threads::pool().stats().dispatches;
        for t in warmup..warmup + tokens {
            let ids = self.draw(arm.n_host * arm.rows, round, t);
            let t0 = Instant::now();
            self.token(arm, &ids, t, &mut tally)?;
            ms.push(t0.elapsed().as_secs_f64() * 1e3);
        }
        let d1 = threads::pool().stats().dispatches;
        let (min1, maj1) = faults()?;
        let resident_after = self.pages.resident()?;
        Ok(ArmRound {
            ms,
            resident_before,
            resident_after,
            minflt: min1 - min0,
            majflt: maj1 - maj0,
            dispatches: d1 - d0,
            tally,
        })
    }
}

/// Minimum, mean and maximum.
fn spread(v: &[f64]) -> (f64, f64, f64) {
    let min = v.iter().copied().fold(f64::INFINITY, f64::min);
    let max = v.iter().copied().fold(0.0, f64::max);
    (min, v.iter().sum::<f64>() / v.len() as f64, max)
}

/// `--time` (see the module doc): every round, every arm, a line each, then
/// one summary line per arm.
fn time(bench: &Bench<'_>, opts: &TimeOpts, lease: bool) -> Result<(), BenchError> {
    let threads = threads::pool().threads();
    let total = bench.pages.total();
    let n = opts.arms.len();
    let target_ms = opts.seconds * 1e3 / opts.rounds as f64;
    let mut tokens: Vec<Option<usize>> = vec![None; n];
    let mut rounds: Vec<Vec<ArmRound>> = (0..n).map(|_| Vec::new()).collect();
    for round in 0..opts.rounds {
        for i in 0..n {
            let a = (i + round) % n;
            let arm = opts.arms[a];
            let r = bench.round(arm, round, opts.warmup, tokens[a], target_ms)?;
            tokens[a] = Some(r.ms.len());
            let bytes = arm.bytes_per_token(&bench.layers);
            let (min, mean, max) = spread(&r.ms);
            let ok =
                lease && r.resident_before == total && r.resident_after == total && r.majflt == 0;
            println!(
                "time threads={threads} round={}/{} arm={} tokens={} warmup={} ms_min={min:.3} ms_mean={mean:.3} ms_max={max:.3} \
                 host_bytes_per_token={bytes} gbps_mean={:.2} gbps_best={:.2} dispatches_per_token={:.2} expected={} \
                 minflt={} majflt={} resident_before={}/{total} resident_after={}/{total} lease={} admissible={}",
                round + 1,
                opts.rounds,
                arm.label(),
                r.ms.len(),
                opts.warmup,
                bytes as f64 / mean / 1e6,
                bytes as f64 / min / 1e6,
                r.dispatches as f64 / r.ms.len() as f64,
                arm.dispatches_per_token(&bench.layers),
                r.minflt,
                r.majflt,
                r.resident_before,
                r.resident_after,
                if lease { "held" } else { "none" },
                if ok { "yes" } else { "no" }
            );
            for d in &r.tally.rows {
                let us = d.ns as f64 / d.count as f64 / 1e3;
                println!(
                    "dispatch threads={threads} round={}/{} arm={} kind={} bytes={} count={} us_mean={us:.2} gbps={:.2}",
                    round + 1,
                    opts.rounds,
                    arm.label(),
                    d.kind,
                    d.bytes,
                    d.count,
                    d.bytes as f64 / us / 1e3
                );
            }
            rounds[a].push(r);
        }
    }
    for (arm, rs) in opts.arms.iter().zip(&rounds) {
        let all: Vec<f64> = rs.iter().flat_map(|r| r.ms.iter().copied()).collect();
        let (min, mean, _) = spread(&all);
        let means: Vec<String> = rs
            .iter()
            .map(|r| format!("{:.3}", spread(&r.ms).1))
            .collect();
        let ok = lease
            && rs
                .iter()
                .all(|r| r.resident_before == total && r.resident_after == total && r.majflt == 0);
        let bytes = arm.bytes_per_token(&bench.layers);
        println!(
            "summary threads={threads} arm={} rounds={} tokens={} ms_min={min:.3} ms_mean={mean:.3} round_means=[{}] \
             gbps_mean={:.2} gbps_best={:.2} admissible={}",
            arm.label(),
            rs.len(),
            all.len(),
            means.join(","),
            bytes as f64 / mean / 1e6,
            bytes as f64 / min / 1e6,
            if ok { "yes" } else { "no" }
        );
    }
    Ok(())
}

/// The model file: `$BLOOMERY_REF_MODEL`, which tools/box.sh exports from the
/// model profile (empty counts as unset).
fn model_path() -> Result<String, BenchError> {
    std::env::var("BLOOMERY_REF_MODEL")
        .ok()
        .filter(|p| !p.is_empty())
        .ok_or_else(|| {
            "BLOOMERY_REF_MODEL unset — run through tools/box.sh with BLOOMERY_MODEL=deepseek41, \
             or the just recipes"
                .into()
        })
}

fn run(mode: Mode) -> Result<(), BenchError> {
    let path = model_path()?;
    let split = Split::open(&path)?;
    let meta = Meta::read(&split)?;
    let layers = build_layers(&split, &meta)?;
    print_table(&path, &split, &meta, &layers);
    let past_used = match &mode {
        Mode::Time(opts) => opts.arms.iter().find(|a| a.n_host > meta.n_used).copied(),
        Mode::Check => None,
    };
    if let Some(a) = past_used {
        return Err(format!(
            "arm {}: the file routes {} experts per token",
            a.label(),
            meta.n_used
        )
        .into());
    }
    // The bench owns its main thread, so it takes the dispatcher's cpu slot, as
    // bloomery-decode does; `BLOOMERY_PIN_MAIN=0` leaves it floating.
    let pin_main = std::env::var("BLOOMERY_PIN_MAIN").map_or(true, |v| v != "0");
    let pinned = pin_main && threads::pool().pin_caller();
    println!(
        "v41host pool threads={} pinned_caller={pinned} pin_failed={}",
        threads::pool().threads(),
        threads::pool().pin_failed()
    );
    print_affinity()?;
    let pages = Pages::of(&layers)?;
    let total = pages.total();
    let before = pages.resident()?;
    let t0 = Instant::now();
    pages.populate()?;
    let populate_s = t0.elapsed().as_secs_f64();
    let after = pages.resident()?;
    println!(
        "v41host working_set experts_per_layer={WORKING_SET} bytes={} pages={total} resident_at_open={before} \
         populate_s={populate_s:.2} resident_populated={after} ({:.3} %)",
        pages.bytes(),
        after as f64 * 100.0 / total as f64
    );
    print_mem()?;
    let bench = Bench {
        layers,
        xs: activations(meta.embd),
        pages,
    };
    match mode {
        Mode::Check => check(&bench.layers, meta.n_used, &bench.xs, true),
        Mode::Time(opts) => {
            let lease = std::env::var("BLOOMERY_HOST_LEASE").is_ok_and(|v| v == "1");
            if !lease {
                println!(
                    "[not under lease] — run through tools/ref/host-rate.sh; these numbers are not admissible"
                );
            }
            check(&bench.layers, meta.n_used, &bench.xs, false)?;
            time(&bench, &opts, lease)
        }
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mode = match parse_args(&args) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("bench_v41_host: {e}\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    match run(mode) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("bench_v41_host: {e}");
            ExitCode::FAILURE
        }
    }
}
