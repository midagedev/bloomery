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
//! The read shapes run no kernel. Per layer they issue the engine shape's two
//! dispatches (every expert's gate and up, then every down), cut into lanes
//! the way `ops`'s row dispatch cuts a one-column group — cumulative row
//! bytes, one lane per pool participant — and each participant reads its
//! lane's rows whole, with wide loads folded into a sink printed once: no
//! stealing, no activation stage, no output. `read-mmap` reads the file
//! mapping every other shape reads. `read-thp` reads an anonymous copy of the
//! whole working set, made before the working set is paged in, in a mapping
//! advised `MADV_HUGEPAGE` before its first touch, then `MADV_COLLAPSE`d for
//! the pages a fault left small; the start-up line gives the kernel's
//! transparent-huge-page mode, the copy's size against `MemAvailable`, and
//! the `AnonHugePages` the copy had after the copy and after the collapse
//! (a refused collapse is printed, not fatal). Mapping against
//! anonymous huge pages, same bytes, same dispatches: the pair separates what
//! the mapping costs from what the dispatch costs.
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
//! the expert and the matrix, and the run exits non-zero. On the same layers
//! the union shape runs two rows — the checked token's experts, and the same
//! set shifted by one with one expert of its own — and each row's output
//! must equal, bit for bit, the list-order sum of the engine shape's downs
//! for that row at the union shape's weights. Where the working set was
//! repacked for the `unionr8` shape (every checked layer under `--check`,
//! every layer when a `--time` arm is `unionr8`), the checked layers also run
//! one token of each of [`R8_CHECKS`] through the `union`, `unionr8` and
//! `unionq` shapes, whose outputs must be equal bit for bit, and `--check`
//! runs the pre-timing arm check below for the arms of [`R8_CHECK_ARMS`] on
//! the checked layers.
//!
//! `--time` (lead-only, under `tools/ref/host-rate.sh`, which owns the lease,
//! the witnesses and the thread sweep): the check first, refusing to time if
//! it fails; then `--rounds` rounds of every arm, the arm order rotated each
//! round. Per arm and round, `--warmup` untimed tokens, then as many timed
//! tokens as fill the arm's share of `--seconds` at the first warm-up's pace;
//! later rounds keep that count. Every dispatch is timed on its own too, by
//! kind and weight bytes: bytes per dispatch is the variable this bench is
//! for.
//!
//! A multi-row arm's rows read disjoint experts unless the arm names a union
//! ratio (`u<r>` on the arm, or `--union r` for every multi-row arm without
//! one): then each layer draws `round(r × n_host × rows)` distinct working-set
//! experts and the rows walk that pool in turn, `n_host` consecutive entries
//! each, so the rows share experts the way consecutive positions of a verify
//! pass do and every drawn expert is read by at least one row. The dispatches
//! still stream every row's slots; `union_bytes_per_token` is the file bytes
//! a pass that read each distinct expert once would read.
//!
//! The `union` shape is that pass: per layer one
//! `moe::HostLayer::experts_union_into` over the `rows` columns and their
//! lists (each row's `n_host` working-set experts at weight `1 / n_host`),
//! which reads each distinct expert once and ends in every column's weighted
//! sum. Its layers are built with a SwiGLU limit of 0, which clamps nothing,
//! so its combine is the plain one the other shapes run. The call is timed
//! whole, as one `union` dispatch entry whose bytes are the layer's distinct
//! experts; its pool dispatches are `2 · ceil(union / moe::UNION_CHUNK)` per
//! layer.
//!
//! The `unionr8` shape is the same call with gate and up through the
//! row-lane Q3_K tile (`qdot::dot_q3k_r8_cols`): the working set's gates and
//! ups are repacked once at start-up (`qdot::repack_q3k_r8`, into anonymous
//! memory, before the working set is paged in), and per layer the union's
//! plan, chunks and down dispatch run unchanged. Its gate and up dispatch
//! hands out (expert, 8-row group) units off one counter, and a call of at
//! most `ops::DEFER_MAX_COLS` columns quantizes `x` as claims inside that
//! dispatch; a wider one quantizes it over the pool first, as the union call
//! does. Each distinct expert keeps its own down block, so no chunk copies
//! its downs aside. The `unionq` shape is that call with the column tile on
//! the file's rows (`qdot::dot_row_cols` row by row, as the union call's row
//! dispatch runs it): `unionr8` against `unionq` is the kernel alone,
//! `unionq` against `union` the dispatch. Before timing, one token of every
//! `unionr8` and `unionq` arm runs through its shape and the union shape on
//! every layer, and their outputs must be equal bit for bit.

use std::ffi::{c_int, c_void};
use std::ops::Range;
use std::process::ExitCode;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

use gguf::{GgmlType, Gguf, Split, TensorInfo, dequant_row};
use model::ModelError;
use model::moe::{
    EXPERTS_INTO_MAX, HostLayer, HostLayerSpec, UNION_CHUNK, UNION_MAX_COLS, UnionScratch,
    expert_view,
};
use model::ops::{
    DEFER_MAX_COLS, GroupInput, ShardTensor, Tensor2, Weight, matmul_q, matmul_q_group_cols_into,
    matmul_q_group_into, matmul_q_group_swiglu, matmul_q_group_swiglu_into,
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
/// Columns up to which the union call sums on the caller (moe's own bound,
/// private there); the `unionr8` shape splits a wider sum as it does.
const UNION_INLINE_COLS: usize = 8;
/// The arms `--check` runs the pre-timing arm check for, on the checked
/// layers: the union sitting's shapes.
const R8_CHECK_ARMS: &str =
    "unionr8:4x8u0.125,unionq:4x8u0.125,unionr8:4x16u0.0625,unionq:4x16u0.0625";
/// The `(rows, n_host, distinct)` tokens the `unionr8` check runs on every
/// checked layer: columns per expert 1–2, 3, 3–4 over three chunks, 6, 7–8,
/// 8, 10 (a run of 8 and one of 2) and 16 (two runs of 8).
const R8_CHECKS: [(usize, usize, usize); 8] = [
    (2, 6, 7),
    (5, 6, 10),
    (13, 6, 24),
    (6, 6, 6),
    (11, 6, 9),
    (8, 4, 4),
    (20, 3, 6),
    (16, 4, 4),
];
/// Pool dispatches of the engine shape per layer: the gate+up group and the
/// down group.
const ENGINE_DISPATCHES: usize = 2;

const USAGE: &str = "usage: bench_v41_host --check
       bench_v41_host --time [--rounds N] [--seconds S] [--warmup W] [--arms A,B,...]
       bench_v41_host --time ... [--union R]
  an arm is <engine|engine-sep|union|unionr8|unionq|per-matrix|read-mmap|read-thp>:<n_host>[x<rows>[u<r>]]; the default is engine:6,engine:5,engine:3,per-matrix:6
  union: one union call per layer over the rows (rows <= 512, n_host <= 8)
  unionr8: the union call with gate and up through the row-lane Q3_K tile (same bounds)
  unionq: the unionr8 call with the union call's column tile (its control: same dispatch)
  u<r> (or --union R for every multi-row arm without its own): the rows share experts, the layer's distinct experts
  are round(r x n_host x rows), 1/rows <= r <= 1
  the model is $BLOOMERY_REF_MODEL (tools/box.sh exports it from the deepseek41 profile)";

// The page bookkeeping's libc calls; `std` already links libc on this target.
unsafe extern "C" {
    fn mincore(addr: *mut c_void, length: usize, vec: *mut u8) -> c_int;
    fn madvise(addr: *mut c_void, length: usize, advice: c_int) -> c_int;
    fn mmap(
        addr: *mut c_void,
        length: usize,
        prot: c_int,
        flags: c_int,
        fd: c_int,
        offset: i64,
    ) -> *mut c_void;
    fn munmap(addr: *mut c_void, length: usize) -> c_int;
    safe fn getpagesize() -> c_int;
}

/// `MADV_WILLNEED` in the Linux UAPI (`asm-generic/mman-common.h`).
const MADV_WILLNEED: c_int = 3;
/// `MADV_HUGEPAGE`, same header.
const MADV_HUGEPAGE: c_int = 14;
/// `MADV_COLLAPSE`, same header: collapse a range into huge pages now.
const MADV_COLLAPSE: c_int = 25;
/// `PROT_READ | PROT_WRITE` (`asm-generic/mman-common.h`).
const PROT_RW: c_int = 0x1 | 0x2;
/// `MAP_PRIVATE | MAP_ANONYMOUS` (`linux/mman.h`, `asm-generic/mman-common.h`).
const MAP_PRIVATE_ANON: c_int = 0x02 | 0x20;
/// A transparent huge page on x86-64: one PMD entry.
const HUGE_PAGE: usize = 2 << 20;
/// Bytes compared at each end of every matrix when the copy is verified.
const VERIFY_EDGE: usize = 4096;

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

/// The engine shapes' output blocks, taken once and reused by every token
/// so no timed token takes a block: `gu` holds gate and up interleaved per
/// expert, `down` and `par` one block per expert. Sized for `slots` experts;
/// a layer with fewer uses the prefix. Every cell a dispatch reads is one it
/// wrote in the same token.
struct Blocks {
    gu: Vec<Tensor2>,
    down: Vec<Tensor2>,
    par: Vec<Tensor2>,
    /// The union shape's, `None` for the others.
    union: Option<UnionBlocks>,
    /// The `unionr8` shape's, `None` for the others.
    r8: Option<R8Blocks>,
}

/// The union shape's blocks, made before the round's first token: activation
/// block `o` holds the `rows` columns `(o + i) % N_X` a token reads at offset
/// `o`, then the call's scratch (made for `rows` columns), its `embd × rows`
/// output and the rows' list entries.
struct UnionBlocks {
    xs: Vec<Tensor2>,
    scratch: UnionScratch,
    out: Vec<f32>,
    lists: Vec<[(u32, f32); EXPERTS_INTO_MAX]>,
}

impl UnionBlocks {
    fn new(xs: &[Tensor2], rows: usize, embd: usize, ff: usize) -> Result<UnionBlocks, ModelError> {
        let xs = (0..N_X)
            .map(|o| {
                let mut data = Vec::with_capacity(embd * rows);
                for i in 0..rows {
                    data.extend_from_slice(&xs[(o + i) % N_X].data);
                }
                Tensor2::from_vec(embd, rows, data)
            })
            .collect();
        Ok(UnionBlocks {
            xs,
            scratch: UnionScratch::new(embd, ff, rows)?,
            out: vec![0.0; embd * rows],
            lists: vec![[(0, 0.0); EXPERTS_INTO_MAX]; rows],
        })
    }
}

impl Blocks {
    /// Blocks for `slots` experts of `layers`, whose matrices must all have
    /// the first layer's row counts.
    fn new(layers: &[Layer<'_>], slots: usize) -> Result<Blocks, BenchError> {
        let first = layers.first().ok_or("no layers")?;
        let rows = |l: &Layer<'_>| [GATE, UP, DOWN].map(|m| l.weight(0, m).n());
        let [n_gate, n_up, n_down] = rows(first);
        if n_gate != n_up || layers.iter().any(|l| rows(l) != [n_gate, n_up, n_down]) {
            return Err("expert matrices differ in row count across layers".into());
        }
        let take = |n: usize, count: usize| (0..count).map(|_| Tensor2::scratch(n, 1)).collect();
        Ok(Blocks {
            gu: take(n_gate, 2 * slots),
            down: take(n_down, slots),
            par: take(n_gate, slots),
            union: None,
            r8: None,
        })
    }
}

/// The union shape: one `HostLayer::experts_union_into` over the token's
/// rows, row `i` listing its `n_host` slots' experts at weight `1 / n_host`.
/// Tallied whole, under the bytes of the layer's distinct experts.
#[allow(clippy::too_many_arguments)]
fn union_layer(
    host: &HostLayer,
    split: &Split,
    l: &Layer<'_>,
    slots: &[usize],
    n_host: usize,
    distinct: usize,
    u: &mut UnionBlocks,
    x: usize,
    tally: &mut Tally,
) -> Result<(), ModelError> {
    let w = 1.0 / n_host as f32;
    for (row, run) in u.lists.iter_mut().zip(slots.chunks_exact(n_host)) {
        for (entry, &s) in row.iter_mut().zip(run) {
            *entry = (l.experts[s].id as u32, w);
        }
    }
    let lists: Vec<&[(u32, f32)]> = u.lists.iter().map(|r| &r[..n_host]).collect();
    let t0 = Instant::now();
    host.experts_union_into(split, &u.xs[x], &lists, &mut u.out, &mut u.scratch)?;
    tally.add("union", distinct as u64 * l.expert_bytes(), t0);
    Ok(())
}

/// One layer's gates and ups in `qdot::repack_q3k_r8`'s layout, `[gate, up]`
/// per working-set slot.
type R8Layer = Vec<[Vec<u8>; 2]>;

/// The row-lane layout of every working-set expert's gate and up for the
/// layers in `which`, `None` for the rest. A gate or up that is not Q3_K is
/// refused by name. Setup, not timed: the matrices are repacked on scoped
/// threads, one per pool participant, into anonymous memory.
fn repack_r8(layers: &[Layer<'_>], which: &[usize]) -> Result<Vec<Option<R8Layer>>, BenchError> {
    let mut set: Vec<Option<R8Layer>> = layers.iter().map(|_| None).collect();
    for &li in which {
        let l = &layers[li];
        for m in [GATE, UP] {
            let ty = l.view(0, m).ty;
            if ty != GgmlType::Q3_K {
                return Err(format!(
                    "unionr8: layer {} {} is {ty:?}; the row-lane tile takes Q3_K",
                    l.index, MATRIX[m]
                )
                .into());
            }
        }
        set[li] = Some(
            (0..l.experts.len())
                .map(|s| [GATE, UP].map(|m| vec![0u8; l.weight(s, m).bytes().len()]))
                .collect(),
        );
    }
    let mut jobs: Vec<(Weight<'_>, &mut [u8])> = Vec::new();
    for (li, slot) in set.iter_mut().enumerate() {
        let Some(layer) = slot else { continue };
        for (s, mats) in layer.iter_mut().enumerate() {
            for (m, dst) in [GATE, UP].into_iter().zip(mats.iter_mut()) {
                jobs.push((layers[li].weight(s, m), dst.as_mut_slice()));
            }
        }
    }
    let per = jobs.len().div_ceil(threads::pool().threads()).max(1);
    std::thread::scope(|scope| {
        let workers: Vec<_> = jobs
            .chunks_mut(per)
            .map(|chunk| {
                scope.spawn(move || {
                    chunk
                        .iter_mut()
                        .try_for_each(|(w, dst)| qdot::repack_q3k_r8(w.bytes(), w.n(), w.k(), dst))
                })
            })
            .collect();
        workers.into_iter().try_for_each(|h| {
            h.join()
                .map_err(|_| BenchError::from("unionr8: a repack thread panicked"))?
                .map_err(BenchError::from)
        })
    })?;
    Ok(set)
}

/// A raw pointer one pool dispatch's participants share; each construction
/// site says which cells each participant alone writes. Read it through
/// [`Cells::ptr`], so a closure captures the wrapper, not the bare pointer.
#[derive(Clone, Copy)]
struct Cells<T>(*mut T);

impl<T> Cells<T> {
    fn ptr(self) -> *mut T {
        self.0
    }
}

// SAFETY: the pointer crosses into one pool dispatch only, which ends before
// its owner is touched again, and the construction sites partition the cells
// among the participants.
unsafe impl<T> Send for Cells<T> {}
// SAFETY: as for `Send` — no cell is written by two participants.
unsafe impl<T> Sync for Cells<T> {}

/// `Tensor2::set_cols`, private to `model`: `ne1` columns inside the block's
/// capacity, without a take; a narrower block keeps its storage.
fn narrow(t: &mut Tensor2, ne1: usize) {
    let n = t.ne0 * ne1;
    assert!(
        n <= t.data.capacity(),
        "narrow: {ne1} columns of {} past the block's capacity {}",
        t.ne0,
        t.data.capacity()
    );
    t.data.resize(n, 0.0);
    t.ne1 = ne1;
}

/// A `unionr8` call's plan, the union call's: the distinct experts in
/// ascending id with their working-set slots, and per expert the columns that
/// list it, ascending, each once (`cols[off[d]..off[d + 1]]`). Storage made
/// with the blocks, so a build within their columns allocates nothing.
struct R8Plan {
    ids: Vec<u32>,
    slots: Vec<usize>,
    off: Vec<usize>,
    cols: Vec<usize>,
    pairs: Vec<(u32, usize)>,
}

impl R8Plan {
    fn with_room(cols: usize) -> R8Plan {
        let slots = cols * EXPERTS_INTO_MAX;
        R8Plan {
            ids: Vec::with_capacity(slots),
            slots: Vec::with_capacity(slots),
            off: Vec::with_capacity(slots + 1),
            cols: Vec::with_capacity(slots),
            pairs: Vec::with_capacity(slots),
        }
    }

    /// The plan of the first `n` entries of every list; an expert outside
    /// `l`'s working set is the named error.
    fn build(
        &mut self,
        lists: &[[(u32, f32); EXPERTS_INTO_MAX]],
        n: usize,
        l: &Layer<'_>,
    ) -> Result<(), BenchError> {
        self.pairs.clear();
        for (j, list) in lists.iter().enumerate() {
            self.pairs.extend(list[..n].iter().map(|&(e, _)| (e, j)));
        }
        self.pairs.sort_unstable();
        self.ids.clear();
        self.slots.clear();
        self.off.clear();
        self.cols.clear();
        for (i, &(e, j)) in self.pairs.iter().enumerate() {
            let prev = i.checked_sub(1).map(|p| self.pairs[p]);
            if prev.is_none_or(|(pe, _)| pe != e) {
                let slot = l
                    .experts
                    .iter()
                    .position(|x| x.id == e as usize)
                    .ok_or_else(|| {
                        format!(
                            "unionr8: expert {e} is outside layer {}'s working set",
                            l.index
                        )
                    })?;
                self.ids.push(e);
                self.slots.push(slot);
                self.off.push(self.cols.len());
            }
            // A list that names an expert twice gives its column once.
            if prev != Some((e, j)) {
                self.cols.push(j);
            }
        }
        self.off.push(self.cols.len());
        Ok(())
    }

    fn n(&self) -> usize {
        self.ids.len()
    }

    fn cols(&self, d: usize) -> &[usize] {
        &self.cols[self.off[d]..self.off[d + 1]]
    }

    /// The distinct index of an expert the lists name.
    fn index_of(&self, e: u32) -> usize {
        self.ids
            .binary_search(&e)
            .expect("every listed expert is in the plan")
    }
}

/// The `unionr8` shape's blocks, made before the round's first token: the
/// union shape's activation blocks, the call's quantized columns, one
/// chunk's gate, up and combine blocks, a down block per distinct expert,
/// the output and the rows' lists.
struct R8Blocks {
    xs: Vec<Tensor2>,
    xq: Vec<u8>,
    gate_up: [Tensor2; 2 * UNION_CHUNK],
    pars: [Tensor2; UNION_CHUNK],
    downs: Vec<Tensor2>,
    out: Vec<f32>,
    plan: R8Plan,
    lists: Vec<[(u32, f32); EXPERTS_INTO_MAX]>,
}

impl R8Blocks {
    /// Blocks for calls over `rows` columns and at most `distinct` experts.
    fn new(xs: &[Tensor2], rows: usize, embd: usize, ff: usize, distinct: usize) -> R8Blocks {
        R8Blocks {
            xs: (0..N_X)
                .map(|o| {
                    let mut data = Vec::with_capacity(embd * rows);
                    for i in 0..rows {
                        data.extend_from_slice(&xs[(o + i) % N_X].data);
                    }
                    Tensor2::from_vec(embd, rows, data)
                })
                .collect(),
            xq: vec![0u8; rows * qdot::col_bytes(GgmlType::Q3_K, embd)],
            gate_up: std::array::from_fn(|_| Tensor2::zeros(ff, rows)),
            pars: std::array::from_fn(|_| Tensor2::zeros(ff, rows)),
            downs: (0..distinct).map(|_| Tensor2::zeros(embd, rows)).collect(),
            out: vec![0.0; embd * rows],
            plan: R8Plan::with_room(rows),
            lists: vec![[(0, 0.0); EXPERTS_INTO_MAX]; rows],
        }
    }
}

/// A call's activation columns quantized as claims on the first gate-and-up
/// dispatch's participants, as the union call claims a narrow slot: each
/// takes columns off `next` until none is left, then waits for `done` to
/// count them all. A claimer that unwinds raises `failed`, and the waiters
/// return instead of spinning on a count that will not come.
struct QuantClaims {
    next: AtomicUsize,
    done: AtomicUsize,
    failed: AtomicBool,
}

/// Raises its flag when dropped while its thread unwinds.
struct UnwindFlag<'a>(&'a AtomicBool);

impl Drop for UnwindFlag<'_> {
    fn drop(&mut self) {
        if std::thread::panicking() {
            self.0.store(true, Ordering::Release);
        }
    }
}

impl QuantClaims {
    fn new() -> QuantClaims {
        QuantClaims {
            next: AtomicUsize::new(0),
            done: AtomicUsize::new(0),
            failed: AtomicBool::new(false),
        }
    }

    /// Claims and quantizes columns of `x` into `xq` (`cb` bytes each), then
    /// waits for every column; `false` when a claimer unwound (the pool
    /// rethrows its panic at the caller).
    fn run(&self, x: &Tensor2, xq: Cells<u8>, cb: usize) -> bool {
        let k = x.ne1;
        loop {
            let c = self.next.fetch_add(1, Ordering::Relaxed);
            if c >= k {
                break;
            }
            let _flag = UnwindFlag(&self.failed);
            // SAFETY: column `c` was claimed by this participant alone
            // (`fetch_add`) and lies inside `xq`'s `k · cb` bytes; no reader
            // forms a slice before `done` counts it (Release below, Acquire in
            // the wait).
            let dst = unsafe { std::slice::from_raw_parts_mut(xq.ptr().add(c * cb), cb) };
            qdot::quantize_col(GgmlType::Q3_K, x.col(c), dst);
            self.done.fetch_add(1, Ordering::Release);
        }
        while self.done.load(Ordering::Acquire) < k {
            if self.failed.load(Ordering::Acquire) {
                return false;
            }
            std::hint::spin_loop();
        }
        true
    }
}

/// The tile a `unionr8`-shaped call runs gate and up through.
#[derive(Clone, Copy)]
enum GateUpTile<'a> {
    /// The row-lane tile over the layer's repacked working set (`unionr8`).
    RowLane(&'a [[Vec<u8>; 2]]),
    /// The column tile over the file's rows, row by row and run by run as the
    /// union call's row dispatch walks them (`unionq`): the same dispatch as
    /// `unionr8`, so the two arms differ by the kernel alone.
    Column,
}

/// The `unionr8` and `unionq` shapes: the union shape's call for the token's
/// rows, row `i` listing its `n_host` slots' experts at weight `1 / n_host`
/// (see [`union_r8_call`]). Tallied whole, under the bytes of the layer's
/// distinct experts.
#[allow(clippy::too_many_arguments)]
fn union_r8_layer(
    host: &HostLayer,
    split: &Split,
    l: &Layer<'_>,
    tile: GateUpTile<'_>,
    slots: &[usize],
    n_host: usize,
    distinct: usize,
    u: &mut R8Blocks,
    x: usize,
    tally: &mut Tally,
) -> Result<(), BenchError> {
    let w = 1.0 / n_host as f32;
    for (row, run) in u.lists.iter_mut().zip(slots.chunks_exact(n_host)) {
        for (entry, &s) in row.iter_mut().zip(run) {
            *entry = (l.experts[s].id as u32, w);
        }
    }
    let t0 = Instant::now();
    union_r8_call(host, split, l, tile, x, n_host, u)?;
    let kind = match tile {
        GateUpTile::RowLane(_) => "unionr8",
        GateUpTile::Column => "unionq",
    };
    tally.add(kind, distinct as u64 * l.expert_bytes(), t0);
    Ok(())
}

/// `HostLayer::experts_union_into` over activation block `xi` and the
/// blocks' lists (their first `n_host` entries), with gate and up through
/// `tile` ([`r8_gate_up`]): the same plan, the same chunks of `UNION_CHUNK`
/// distinct experts in ascending id, the same down dispatch
/// (`ops::matmul_q_group_cols_into` over the combines, the layer's clamp) and
/// each column's sum from zero in its list order — so its output is the
/// union call's, bit for bit, when the tile is `dot_row`'s. Allocates
/// nothing.
fn union_r8_call(
    host: &HostLayer,
    split: &Split,
    l: &Layer<'_>,
    tile: GateUpTile<'_>,
    xi: usize,
    n_host: usize,
    u: &mut R8Blocks,
) -> Result<(), BenchError> {
    let R8Blocks {
        xs,
        xq,
        gate_up,
        pars,
        downs,
        out,
        plan,
        lists,
    } = u;
    let x = &xs[xi];
    let (embd, k) = (x.ne0, x.ne1);
    plan.build(lists, n_host, l)?;
    if plan.n() > downs.len() {
        return Err(format!(
            "unionr8: {} distinct experts, the blocks hold {}",
            plan.n(),
            downs.len()
        )
        .into());
    }
    let cb = qdot::col_bytes(GgmlType::Q3_K, embd);
    let claims = QuantClaims::new();
    let narrow_call = k <= DEFER_MAX_COLS;
    if !narrow_call && plan.n() > 0 {
        let dst = Cells(xq.as_mut_ptr());
        // SAFETY (construction site): `xq` holds `k · cb` bytes, borrowed for
        // the whole dispatch; each participant writes only the columns of its
        // own chunk of `0..k`, the chunks partition them, and the join
        // publishes the writes.
        threads::pool().for_each_chunk(k, |cols| {
            for c in cols {
                // SAFETY: column `c` is in this participant's chunk — see the construction site.
                let col = unsafe { std::slice::from_raw_parts_mut(dst.ptr().add(c * cb), cb) };
                qdot::quantize_col(GgmlType::Q3_K, x.col(c), col);
            }
        });
    }
    let limit = host.swiglu_limit();
    let down_stack = host.stacks()[DOWN];
    let n_chunks = plan.n().div_ceil(UNION_CHUNK);
    let per = if n_chunks == 0 {
        0
    } else {
        plan.n().div_ceil(n_chunks)
    };
    for ch in 0..n_chunks {
        let (a, b) = (ch * per, ((ch + 1) * per).min(plan.n()));
        let n = b - a;
        for i in 0..n {
            let m = plan.cols(a + i).len();
            narrow(&mut gate_up[2 * i], m);
            narrow(&mut gate_up[2 * i + 1], m);
            narrow(&mut pars[i], m);
            narrow(&mut downs[a + i], m);
        }
        let claim = (narrow_call && ch == 0).then_some(&claims);
        let empty: &[u8] = &[];
        let mut mats = [[empty; 2]; UNION_CHUNK];
        for (i, m) in mats.iter_mut().enumerate().take(n) {
            let s = plan.slots[a + i];
            *m = match tile {
                GateUpTile::RowLane(r8) => [r8[s][0].as_slice(), r8[s][1].as_slice()],
                GateUpTile::Column => [l.weight(s, GATE).bytes(), l.weight(s, UP).bytes()],
            };
        }
        let row_lane = matches!(tile, GateUpTile::RowLane(_));
        r8_gate_up(
            plan,
            &mats[..n],
            row_lane,
            x,
            xq,
            cb,
            claim,
            a,
            &mut gate_up[..2 * n],
        )?;
        let first = down_stack.expert(split, plan.ids[a] as usize)?;
        let mut down = [first; UNION_CHUNK];
        for (i, d) in down.iter_mut().enumerate().take(n) {
            *d = down_stack.expert(split, plan.ids[a + i] as usize)?;
        }
        let combines: [GroupInput<'_>; UNION_CHUNK] = std::array::from_fn(|i| {
            if i < n {
                GroupInput::SwigluClamp(&gate_up[2 * i], &gate_up[2 * i + 1], limit)
            } else {
                GroupInput::Ready(x)
            }
        });
        matmul_q_group_cols_into(
            "host_down",
            &down[..n],
            &combines[..n],
            &mut downs[a..b],
            &mut pars[..n],
        )?;
    }
    let (plan, downs, lists) = (&*plan, &*downs, &*lists);
    // Column `j`'s weighted sum into `o`, from zero in the order of its list
    // — the union call's `sum_col`.
    let sum_col = |j: usize, o: &mut [f32]| {
        o.fill(0.0);
        for &(e, w) in &lists[j][..n_host] {
            let d = plan.index_of(e);
            let t = plan
                .cols(d)
                .binary_search(&j)
                .expect("a listing column is in its expert's columns");
            for (o, &dv) in o.iter_mut().zip(downs[d].col(t)) {
                *o += w * dv;
            }
        }
    };
    if k <= UNION_INLINE_COLS {
        for (j, o) in out.chunks_exact_mut(embd).enumerate() {
            sum_col(j, o);
        }
    } else {
        let dst = Cells(out.as_mut_ptr());
        // SAFETY (construction site): `out` holds `k · embd` values, borrowed
        // for the whole dispatch; each participant writes only the columns of
        // its own chunk of `0..k`, the chunks partition them, and the join
        // publishes the writes.
        threads::pool().for_each_chunk(k, |cols| {
            for j in cols {
                // SAFETY: column `j` is in this participant's chunk — see the construction site.
                let o = unsafe { std::slice::from_raw_parts_mut(dst.ptr().add(j * embd), embd) };
                sum_col(j, o);
            }
        });
    }
    Ok(())
}

/// One chunk's gates and ups over the pool: unit `u` is distinct expert
/// `a + u / groups`'s 8-row group `u % groups`, gate then up, claimed in
/// order off one counter; `mats[i]` holds the chunk's expert `i`'s gate and
/// up, in the row-lane layout (`row_lane`: each run of up to
/// `qdot::TILE_COLS` of the expert's columns goes to `qdot::dot_q3k_r8_cols`
/// once for the eight rows) or as the file's rows (each of the eight rows
/// through `qdot::dot_row_cols` per run, as the union call's row dispatch
/// does). The values land in `outs[2i]` (gate) and `outs[2i + 1]` (up),
/// already narrowed to the expert's columns. With `claims`, the participants
/// first quantize `x` into `xq`; without, `xq` already holds it. A kernel
/// error stops the unit walk and is returned.
#[allow(clippy::too_many_arguments)]
fn r8_gate_up(
    plan: &R8Plan,
    mats: &[[&[u8]; 2]],
    row_lane: bool,
    x: &Tensor2,
    xq: &mut [u8],
    cb: usize,
    claims: Option<&QuantClaims>,
    a: usize,
    outs: &mut [Tensor2],
) -> Result<(), BenchError> {
    const R8: usize = qdot::Q3K_R8_ROWS;
    let embd = x.ne0;
    let Some(ff) = outs.first().map(|t| t.ne0) else {
        return Ok(());
    };
    let groups = ff / R8;
    let units = mats.len() * groups;
    let next = AtomicUsize::new(0);
    let stop = AtomicBool::new(false);
    let fail: Mutex<Option<qdot::QdotError>> = Mutex::new(None);
    let xq_len = xq.len();
    let xqp = Cells(xq.as_mut_ptr());
    // SAFETY (construction site): `outs[p]` is a `{ff, m}` block of the
    // caller's, borrowed for the whole dispatch; unit `u` writes only rows
    // `8g..8g + 8` of every column of its expert's gate and up, the units
    // partition those cells, each unit is claimed once, and the join
    // publishes the writes. `xq` is written only by the claims, each column by
    // its claimer, and read only after `QuantClaims::run` observed every
    // column (or, without claims, after the fill dispatch's join).
    let dst: [Cells<f32>; 2 * UNION_CHUNK] = std::array::from_fn(|p| {
        Cells(
            outs.get_mut(p)
                .map_or(std::ptr::null_mut(), |t| t.data.as_mut_ptr()),
        )
    });
    // The first failing call's error; the walk stops at the next unit.
    let failed = |e: qdot::QdotError| {
        stop.store(true, Ordering::Relaxed);
        fail.lock().expect("unionr8 error slot").get_or_insert(e);
    };
    threads::pool().for_each_chunk(threads::pool().threads(), |_| {
        if let Some(q) = claims
            && !q.run(x, xqp, cb)
        {
            return;
        }
        // SAFETY: every column is written and published — see the construction site.
        let xq = unsafe { std::slice::from_raw_parts(xqp.ptr().cast_const(), xq_len) };
        while !stop.load(Ordering::Relaxed) {
            let u = next.fetch_add(1, Ordering::Relaxed);
            if u >= units {
                return;
            }
            let (i, g) = (u / groups, u % groups);
            let cols = plan.cols(a + i);
            for (m, w) in mats[i].iter().enumerate() {
                let out = dst[2 * i + m];
                for (t0, run) in (0..)
                    .step_by(qdot::TILE_COLS)
                    .zip(cols.chunks(qdot::TILE_COLS))
                {
                    let acols: [&[u8]; qdot::TILE_COLS] = std::array::from_fn(|t| {
                        let c = run[t.min(run.len() - 1)];
                        &xq[c * cb..(c + 1) * cb]
                    });
                    let acols = &acols[..run.len()];
                    if row_lane {
                        let gb = w.len() / groups;
                        let mut v = [[0.0f32; R8]; qdot::TILE_COLS];
                        let r = qdot::dot_q3k_r8_cols(
                            &w[g * gb..(g + 1) * gb],
                            acols,
                            embd,
                            &mut v[..run.len()],
                        );
                        if let Err(e) = r {
                            return failed(e);
                        }
                        for (t, vals) in v[..run.len()].iter().enumerate() {
                            let at = (t0 + t) * ff + R8 * g;
                            // SAFETY: rows 8g..8g + 8 of this column belong to unit `u` alone — see the construction site.
                            unsafe {
                                std::ptr::copy_nonoverlapping(vals.as_ptr(), out.ptr().add(at), R8);
                            }
                        }
                    } else {
                        let rb = w.len() / ff;
                        for row in R8 * g..R8 * (g + 1) {
                            let mut v = [0.0f32; qdot::TILE_COLS];
                            let src = &w[row * rb..(row + 1) * rb];
                            let r = qdot::dot_row_cols(
                                GgmlType::Q3_K,
                                src,
                                acols,
                                embd,
                                &mut v[..run.len()],
                            );
                            if let Err(e) = r {
                                return failed(e);
                            }
                            for (t, &val) in v[..run.len()].iter().enumerate() {
                                // SAFETY: row `row` of this column belongs to unit `u` alone — see the construction site.
                                unsafe { out.ptr().add((t0 + t) * ff + row).write(val) };
                            }
                        }
                    }
                }
            }
        }
    });
    match fail.into_inner().expect("unionr8 error slot") {
        Some(e) => Err(e.into()),
        None => Ok(()),
    }
}

/// The engine's shape: gate and up of every expert in one group, the
/// weights interleaved per expert as the host tier lays them out, each read
/// from its own shard, then one down group whose inputs are the SwiGLU
/// combines.
///
/// Its outputs land in the prefix of `b`: gate and up of expert `i` in
/// `b.gu[2i]` and `b.gu[2i + 1]`, down and combine in `b.down[i]`, `b.par[i]`.
fn engine_layer(
    l: &Layer<'_>,
    slots: &[usize],
    x: &Tensor2,
    b: &mut Blocks,
    tally: &mut Tally,
) -> Result<(), ModelError> {
    engine_rows_layer(l, slots, &[x], b, tally)
}

/// Probe shape for a k-token step: `xs.len()` rows, row `i` owning the
/// `slots.len() / xs.len()` slots `slots[i * n..(i + 1) * n]` and reading
/// `xs[i]`. Every row's gate and up in one group, every row's down in one
/// group — the dispatches a tier that takes k tokens per call would issue.
fn engine_rows_layer(
    l: &Layer<'_>,
    slots: &[usize],
    xs: &[&Tensor2],
    b: &mut Blocks,
    tally: &mut Tally,
) -> Result<(), ModelError> {
    let n = slots.len();
    let per_row = slots.len() / xs.len();
    let xin: Vec<&Tensor2> = (0..slots.len())
        .flat_map(|j| [xs[j / per_row]; 2])
        .collect();
    let ws: Vec<Weight<'_>> = slots
        .iter()
        .flat_map(|&s| [l.weight(s, GATE), l.weight(s, UP)])
        .collect();
    let t0 = Instant::now();
    matmul_q_group_into("host_gate_up", &ws, &xin, &mut b.gu[..2 * n])?;
    tally.add("gate+up", bytes_of(&ws), t0);
    let dw: Vec<Weight<'_>> = slots.iter().map(|&s| l.weight(s, DOWN)).collect();
    let (pairs, _) = b.gu[..2 * n].as_chunks::<2>();
    let srcs: Vec<GroupInput<'_>> = pairs
        .iter()
        .map(|[g, u]| GroupInput::Swiglu(g, u))
        .collect();
    let t0 = Instant::now();
    matmul_q_group_swiglu_into("host_down", &dw, &srcs, &mut b.down[..n], &mut b.par[..n])?;
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

/// One expert's `[gate, up, down]` bytes, as a read shape reads them.
type ExpertBytes<'a> = [&'a [u8]; 3];

/// Every layer's working set, per slot, as the file mapping holds it.
fn mapped_bytes<'a>(layers: &[Layer<'a>]) -> Vec<Vec<ExpertBytes<'a>>> {
    layers
        .iter()
        .map(|l| {
            (0..l.experts.len())
                .map(|s| [GATE, UP, DOWN].map(|m| l.weight(s, m).bytes()))
                .collect()
        })
        .collect()
}

/// A read shape's layer: the engine shape's two dispatches — every slot's
/// gate and up interleaved, then every down — each reading its matrices'
/// bytes from `set` and nothing else. The tally rows carry the engine
/// shape's kinds and bytes, so the two line up.
fn read_layer(
    l: &Layer<'_>,
    set: &[ExpertBytes<'_>],
    slots: &[usize],
    sink: &AtomicU64,
    tally: &mut Tally,
) {
    let gu: Vec<(&[u8], usize)> = slots
        .iter()
        .flat_map(|&s| [GATE, UP].map(|m| (set[s][m], l.weight(s, m).n())))
        .collect();
    let t0 = Instant::now();
    read_group(&gu, sink);
    tally.add("gate+up", read_bytes(&gu), t0);
    let dn: Vec<(&[u8], usize)> = slots
        .iter()
        .map(|&s| (set[s][DOWN], l.weight(s, DOWN).n()))
        .collect();
    let t0 = Instant::now();
    read_group(&dn, sink);
    tally.add("down", read_bytes(&dn), t0);
}

/// Bytes a read dispatch reads.
fn read_bytes(pairs: &[(&[u8], usize)]) -> u64 {
    pairs.iter().map(|p| p.0.len() as u64).sum()
}

/// One pool dispatch over `pairs` (a matrix's bytes and its row count
/// each): one lane per participant, cut by [`lane_bounds`], read whole and
/// folded into `sink`.
fn read_group(pairs: &[(&[u8], usize)], sink: &AtomicU64) {
    let nlanes = threads::pool().threads();
    let bounds = lane_bounds(pairs, nlanes);
    threads::pool().for_each_chunk(nlanes, |lanes| {
        let mut acc = 0u64;
        for t in lanes {
            acc = acc.wrapping_add(read_rows(pairs, bounds[t]..bounds[t + 1]));
        }
        sink.fetch_add(acc, Ordering::Relaxed);
    });
}

/// The lane cut of `ops`'s row dispatch for a one-column group: over the
/// pairs' concatenated rows, lane `t` starts at the first row where
/// `nlanes` times the running byte cost reaches `t` times the total — the
/// same closed form per pair, so a uniform group is a row-count cut.
fn lane_bounds(pairs: &[(&[u8], usize)], nlanes: usize) -> Vec<usize> {
    let row_cost = |p: &(&[u8], usize)| (p.0.len() / p.1.max(1)) as u64;
    let total_rows: usize = pairs.iter().map(|p| p.1).sum();
    let total_cost: u64 = pairs.iter().map(|p| p.1 as u64 * row_cost(p)).sum();
    let nl = nlanes as u64;
    let mut bounds = vec![0usize; nlanes + 1];
    // `p` the pair the cut is in, `before` the cost ahead of it, `first` its
    // first row.
    let (mut p, mut before, mut first) = (0usize, 0u64, 0usize);
    for (t, bound) in bounds.iter_mut().enumerate().take(nlanes).skip(1) {
        let target = t as u64 * total_cost;
        loop {
            let Some(pair) = pairs.get(p) else {
                *bound = total_rows;
                break;
            };
            let (c, n_p) = (row_cost(pair), pair.1 as u64);
            if c == 0 || (before + n_p * c) * nl < target {
                before += n_p * c;
                first += pair.1;
                p += 1;
                continue;
            }
            let need = target.saturating_sub(before * nl);
            *bound = first + need.div_ceil(c * nl).min(n_p) as usize;
            break;
        }
    }
    bounds[nlanes] = total_rows;
    bounds
}

/// Rows `rows` of the pairs' concatenated rows, every byte read and folded.
fn read_rows(pairs: &[(&[u8], usize)], rows: Range<usize>) -> u64 {
    let mut acc = 0u64;
    let mut first = 0usize;
    for &(bytes, n) in pairs {
        let (lo, hi) = (rows.start.max(first), rows.end.min(first + n));
        if lo < hi {
            let rb = bytes.len() / n;
            acc = acc.wrapping_add(fold_bytes(&bytes[(lo - first) * rb..(hi - first) * rb]));
        }
        first += n;
    }
    acc
}

/// Every byte of `s` read and folded into one word: eight independent 64-bit
/// sums over 64-byte blocks, which the vectorizer turns into full-width
/// vector loads and adds, then the tail. Not inlined, so its code can be
/// found and read in the binary.
#[inline(never)]
fn fold_bytes(s: &[u8]) -> u64 {
    let (blocks, tail) = s.as_chunks::<64>();
    let mut acc = [0u64; 8];
    for b in blocks {
        let (words, _) = b.as_chunks::<8>();
        for (a, w) in acc.iter_mut().zip(words) {
            *a = a.wrapping_add(u64::from_ne_bytes(*w));
        }
    }
    let tail = tail.iter().fold(0u64, |a, &b| a.wrapping_add(u64::from(b)));
    acc.iter().fold(tail, |a, &w| a.wrapping_add(w))
}

/// An anonymous private mapping advised `MADV_HUGEPAGE` before its first
/// touch, holding a copy of every layer's working set from a huge-page
/// boundary on; unmapped on drop.
struct ThpCopy {
    map: *mut c_void,
    map_len: usize,
    /// Bytes from `map` to the first huge-page boundary, where the copy starts.
    lead: usize,
    len: usize,
    /// Per layer, per working-set slot: where `[gate, up, down]` sit.
    at: Vec<Vec<[Range<usize>; 3]>>,
}

impl ThpCopy {
    /// Map, advise, copy every matrix of `mapped` (each from a 64-byte
    /// boundary), verify, and print the start-up line. Refused when
    /// `MemAvailable` cannot hold the copy and the working set's page-cache
    /// pages both (it counts the latter as reclaimable).
    fn make(mapped: &[Vec<ExpertBytes<'_>>]) -> Result<ThpCopy, BenchError> {
        let mut len = 0usize;
        let at: Vec<Vec<[Range<usize>; 3]>> = mapped
            .iter()
            .map(|layer| {
                layer
                    .iter()
                    .map(|e| {
                        e.map(|b| {
                            let start = len.next_multiple_of(64);
                            len = start + b.len();
                            start..len
                        })
                    })
                    .collect()
            })
            .collect();
        let available = mem_available()?;
        let need = 2 * len as u64;
        if available < need {
            return Err(format!(
                "read-thp: the copy is {len} B and the working set's page-cache pages as many; \
                 MemAvailable is {available} B, under the {need} B both take"
            )
            .into());
        }
        let map_len = len.next_multiple_of(HUGE_PAGE) + HUGE_PAGE;
        // SAFETY: a fresh anonymous private mapping with no address hint and no
        // file: it aliases nothing, and a failure comes back as MAP_FAILED.
        let map = unsafe {
            mmap(
                std::ptr::null_mut(),
                map_len,
                PROT_RW,
                MAP_PRIVATE_ANON,
                -1,
                0,
            )
        };
        if map.addr() == usize::MAX {
            return Err(format!("mmap {map_len} B: {}", std::io::Error::last_os_error()).into());
        }
        let lead = map.addr().next_multiple_of(HUGE_PAGE) - map.addr();
        let mut copy = ThpCopy {
            map,
            map_len,
            lead,
            len,
            at: Vec::new(),
        };
        // SAFETY: `lead + len` rounded up to a huge page is at most `map_len`,
        // so the range lies inside the mapping just made; HUGEPAGE changes only
        // how the kernel backs pages not yet touched, and none is.
        let rc = unsafe {
            madvise(
                map.wrapping_byte_add(lead),
                len.next_multiple_of(HUGE_PAGE),
                MADV_HUGEPAGE,
            )
        };
        if rc != 0 {
            return Err(format!("madvise(HUGEPAGE): {}", std::io::Error::last_os_error()).into());
        }
        let t0 = Instant::now();
        let data = copy.data_mut();
        for (layer, ranges) in mapped.iter().zip(&at) {
            for (e, r) in layer.iter().zip(ranges) {
                for (b, r) in e.iter().zip(r) {
                    data[r.clone()].copy_from_slice(b);
                }
            }
        }
        let copy_s = t0.elapsed().as_secs_f64();
        copy.at = at;
        let huge_copied = anon_huge_kb()?;
        let t0 = Instant::now();
        // SAFETY: the range `madvise(HUGEPAGE)` took above. COLLAPSE moves the
        // range's contents into huge pages; no byte of it changes.
        let rc = unsafe {
            madvise(
                map.wrapping_byte_add(lead),
                len.next_multiple_of(HUGE_PAGE),
                MADV_COLLAPSE,
            )
        };
        let collapse = if rc == 0 {
            "ok".to_string()
        } else {
            format!("refused({})", std::io::Error::last_os_error())
        };
        let collapse_s = t0.elapsed().as_secs_f64();
        let checked = copy.verify(mapped)?;
        println!(
            "v41host thp_copy bytes={len} mem_available={available} thp_enabled={} copy_s={copy_s:.2} \
             anon_huge_kb_copied={huge_copied} collapse={collapse} collapse_s={collapse_s:.2} \
             anon_huge_kb={} verified=[{checked}]",
            thp_mode(),
            anon_huge_kb()?
        );
        Ok(copy)
    }

    /// The copy's bytes.
    fn data(&self) -> &[u8] {
        // SAFETY: `lead..lead + len` lies inside the live mapping (see `make`);
        // anonymous pages read as zeros until written, so every byte is
        // initialized; the borrow of `self` keeps the mapping alive and rules
        // out a `data_mut` borrow for as long as the slice lives.
        unsafe { std::slice::from_raw_parts(self.map.cast::<u8>().add(self.lead), self.len) }
    }

    /// The copy's bytes, writable: the same range as `data`.
    fn data_mut(&mut self) -> &mut [u8] {
        // SAFETY: as `data`; the exclusive borrow of `self` makes this the only
        // live view of the range.
        unsafe { std::slice::from_raw_parts_mut(self.map.cast::<u8>().add(self.lead), self.len) }
    }

    /// Every layer's working set, per slot, as the copy holds it.
    fn bytes(&self) -> Vec<Vec<ExpertBytes<'_>>> {
        let d = self.data();
        self.at
            .iter()
            .map(|l| l.iter().map(|r| r.clone().map(|r| &d[r])).collect())
            .collect()
    }

    /// The copy against its source: the first and last [`VERIFY_EDGE`] bytes
    /// of every matrix, and every byte of the first, middle and last layers.
    /// Returns what it compared.
    fn verify(&self, mapped: &[Vec<ExpertBytes<'_>>]) -> Result<String, BenchError> {
        let whole = [0, mapped.len() / 2, mapped.len().saturating_sub(1)];
        for (li, (src, dst)) in mapped.iter().zip(self.bytes()).enumerate() {
            for (s, (a, b)) in src.iter().zip(&dst).enumerate() {
                for m in [GATE, UP, DOWN] {
                    let (a, b) = (a[m], b[m]);
                    let e = VERIFY_EDGE.min(a.len());
                    let same = a.len() == b.len()
                        && if whole.contains(&li) {
                            a == b
                        } else {
                            a[..e] == b[..e] && a[a.len() - e..] == b[b.len() - e..]
                        };
                    if !same {
                        return Err(format!(
                            "read-thp: the copy of layer {li} slot {s} {} differs from the file",
                            MATRIX[m]
                        )
                        .into());
                    }
                }
            }
        }
        Ok(format!(
            "{VERIFY_EDGE} B at both ends of every matrix, every byte of layers {whole:?}"
        ))
    }
}

impl Drop for ThpCopy {
    fn drop(&mut self) {
        // SAFETY: `map` and `map_len` are the mapping `make` got from mmap,
        // unmapped here once; every slice borrowed from `self` is dead.
        let rc = unsafe { munmap(self.map, self.map_len) };
        if rc != 0 {
            eprintln!("munmap: {}", std::io::Error::last_os_error());
        }
    }
}

/// `MemAvailable` from `/proc/meminfo`, in bytes.
fn mem_available() -> Result<u64, BenchError> {
    let s = std::fs::read_to_string("/proc/meminfo")?;
    let kb: u64 = s
        .lines()
        .find_map(|l| l.strip_prefix("MemAvailable:"))
        .ok_or("/proc/meminfo has no MemAvailable")?
        .trim()
        .trim_end_matches("kB")
        .trim()
        .parse()?;
    Ok(kb * 1024)
}

/// The kernel's transparent-huge-page mode: the bracketed word of
/// `/sys/kernel/mm/transparent_hugepage/enabled`.
fn thp_mode() -> String {
    std::fs::read_to_string("/sys/kernel/mm/transparent_hugepage/enabled")
        .ok()
        .and_then(|s| {
            let (_, rest) = s.split_once('[')?;
            Some(rest.split_once(']')?.0.to_string())
        })
        .unwrap_or_else(|| "unreadable".to_string())
}

/// `AnonHugePages` of `/proc/self/smaps_rollup`, in kB.
fn anon_huge_kb() -> Result<u64, BenchError> {
    let s = std::fs::read_to_string("/proc/self/smaps_rollup")?;
    Ok(s.lines()
        .find_map(|l| l.strip_prefix("AnonHugePages:"))
        .ok_or("smaps_rollup has no AnonHugePages")?
        .trim()
        .trim_end_matches("kB")
        .trim()
        .parse()?)
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
/// The union shape against the engine shape on layer `l`: two rows over
/// `xs[li % N_X]` and the next column, row 0 listing `slots`, row 1 the same
/// set shifted by one with one working-set expert neither lists. Each row's
/// output must equal the list-order sum from zero of the engine shape's downs
/// for that row, at weight `1 / n`, bit for bit.
fn check_union(
    c: &mut Checks,
    l: &Layer<'_>,
    host: &HostLayer,
    split: &Split,
    slots: &[usize],
    xs: &[Tensor2],
) -> Result<(), BenchError> {
    let n = slots.len();
    let extra = (0..WORKING_SET)
        .find(|s| !slots.contains(s))
        .ok_or("the working set holds no expert outside the checked token's")?;
    let mut rows: Vec<usize> = slots.to_vec();
    rows.extend(slots[1..].iter().copied());
    rows.push(extra);
    let embd = xs[0].ne0;
    let ff = l.weight(0, GATE).n();
    let mut u = UnionBlocks::new(xs, 2, embd, ff)?;
    let x0 = l.index % N_X;
    let mut tally = Tally::default();
    union_layer(host, split, l, &rows, n, n + 1, &mut u, x0, &mut tally)?;
    let w = 1.0 / n as f32;
    let mut same = true;
    for (i, run) in rows.chunks_exact(n).enumerate() {
        let mut b = Blocks::new(std::slice::from_ref(l), n)?;
        engine_layer(l, run, &xs[(x0 + i) % N_X], &mut b, &mut tally)?;
        let mut want = vec![0.0f32; embd];
        for d in &b.down[..n] {
            for (o, &dv) in want.iter_mut().zip(d.col(0)) {
                *o += w * dv;
            }
        }
        same &= bits_equal(&u.out[i * embd..(i + 1) * embd], &want);
    }
    let line = format!(
        "check layer={} shape=union rows=2 experts={n} distinct={} outputs_bits_equal_engine_sum={same}",
        l.index,
        n + 1
    );
    c.record(format!("layer {} union", l.index), &line, same);
    Ok(())
}

fn check_layer(
    c: &mut Checks,
    l: &Layer<'_>,
    slots: &[usize],
    x: &Tensor2,
) -> Result<(), BenchError> {
    let mut tally = Tally::default();
    let n = slots.len();
    let mut b = Blocks::new(std::slice::from_ref(l), n)?;
    engine_layer(l, slots, x, &mut b, &mut tally)?;
    let pm = per_matrix_layer(l, slots, x, &mut tally)?;
    let gate: Vec<&Tensor2> = b.gu.iter().step_by(2).collect();
    let up: Vec<&Tensor2> = b.gu.iter().skip(1).step_by(2).collect();
    let ours = [gate, up, b.down.iter().collect(), b.par.iter().collect()];
    let theirs = [&pm.gate, &pm.up, &pm.down, &pm.par];
    let same = ours.iter().zip(theirs).all(|(a, b)| {
        a.len() == b.len()
            && a.iter()
                .zip(b.iter())
                .all(|(a, b)| bits_equal(&a.data, &b.data))
    });
    let [gate, up, down, par] = ours;
    let line = format!(
        "check layer={} shape=per-matrix experts={} outputs_bits_equal_engine={same}",
        l.index,
        slots.len()
    );
    c.record(format!("layer {} per-matrix", l.index), &line, same);
    let act_gate = kernel_activation(l.view(slots[0], GATE).ty, &x.data)?;
    let act_up = kernel_activation(l.view(slots[0], UP).ty, &x.data)?;
    for (i, &s) in slots.iter().enumerate() {
        check_weight(c, l, s, GATE, gate[i], &act_gate)?;
        check_weight(c, l, s, UP, up[i], &act_up)?;
        check_swiglu(c, l, s, gate[i], up[i], par[i]);
        let act_down = kernel_activation(l.view(s, DOWN).ty, &par[i].data)?;
        check_weight(c, l, s, DOWN, down[i], &act_down)?;
    }
    Ok(())
}

/// The `unionr8` and `unionq` shapes against the union shape on layer `li`,
/// bit for bit, for one token of `rows` rows walking a pool of `distinct`
/// working-set slots, `n_host` each, as an arm's rows do.
fn check_union_r8(
    c: &mut Checks,
    bench: &Bench<'_>,
    li: usize,
    (rows, n_host, distinct_n): (usize, usize, usize),
) -> Result<(), BenchError> {
    let l = &bench.layers[li];
    let host = bench
        .hosts
        .get(li)
        .ok_or("no host layer to check the unionr8 shape")?;
    let r8 = bench
        .r8
        .get(li)
        .and_then(Option::as_ref)
        .ok_or_else(|| format!("layer {} was not repacked for the unionr8 check", l.index))?;
    let embd = bench.xs[0].ne0;
    let ff = l.weight(0, GATE).n();
    let key = [KEY_CHECK, li as u64, rows as u64, n_host as u64];
    let pool = distinct(&mut Rng::new(&key), distinct_n, WORKING_SET);
    let slots: Vec<usize> = (0..rows * n_host).map(|i| pool[i % distinct_n]).collect();
    let x0 = li % N_X;
    let mut tally = Tally::default();
    let mut u = UnionBlocks::new(&bench.xs, rows, embd, ff)?;
    union_layer(
        host,
        bench.split,
        l,
        &slots,
        n_host,
        distinct_n,
        &mut u,
        x0,
        &mut tally,
    )?;
    for (shape, tile) in [
        ("unionr8", GateUpTile::RowLane(r8)),
        ("unionq", GateUpTile::Column),
    ] {
        let mut v = R8Blocks::new(&bench.xs, rows, embd, ff, distinct_n);
        union_r8_layer(
            host,
            bench.split,
            l,
            tile,
            &slots,
            n_host,
            distinct_n,
            &mut v,
            x0,
            &mut tally,
        )?;
        let same = bits_equal(&u.out, &v.out);
        let line = format!(
            "check layer={} shape={shape} rows={rows} experts={n_host} distinct={distinct_n} \
             outputs_bits_equal_union={same}",
            l.index
        );
        c.record(
            format!("layer {} {shape} rows {rows}", l.index),
            &line,
            same,
        );
    }
    Ok(())
}

/// Before any timing (every layer), and under `--check` (the checked
/// layers): one token of every `unionr8` and `unionq` arm through its shape
/// and the union shape, layer by layer over `layers`, whose outputs must be
/// equal bit for bit; a mismatch is the named error.
fn check_r8_arms(bench: &Bench<'_>, arms: &[Arm], layers: &[usize]) -> Result<(), BenchError> {
    let embd = bench.xs.first().ok_or("no activation columns")?.ne0;
    for arm in arms
        .iter()
        .filter(|a| matches!(a.shape, Shape::UnionR8 | Shape::UnionQ))
    {
        let ids = bench.draw(*arm, 0, 0);
        let mut differ = Vec::new();
        for &l in layers {
            let layer = &bench.layers[l];
            let host = bench
                .hosts
                .get(l)
                .ok_or("no host layer for the unionr8 arm")?;
            let tile = bench.gate_up_tile(arm.shape, l)?;
            let ff = layer.weight(0, GATE).n();
            let x = l % N_X;
            let mut tally = Tally::default();
            let mut u = UnionBlocks::new(&bench.xs, arm.rows, embd, ff)?;
            union_layer(
                host,
                bench.split,
                layer,
                &ids[l],
                arm.n_host,
                arm.union,
                &mut u,
                x,
                &mut tally,
            )?;
            let mut v = R8Blocks::new(&bench.xs, arm.rows, embd, ff, arm.union);
            union_r8_layer(
                host,
                bench.split,
                layer,
                tile,
                &ids[l],
                arm.n_host,
                arm.union,
                &mut v,
                x,
                &mut tally,
            )?;
            if !bits_equal(&u.out, &v.out) {
                differ.push(layer.index);
            }
        }
        println!(
            "check arm={} layers={} outputs_bits_equal_union={}",
            arm.label(),
            layers.len(),
            differ.is_empty()
        );
        if !differ.is_empty() {
            return Err(format!(
                "arm {}: its output differs from the union output on layers {differ:?}",
                arm.label()
            )
            .into());
        }
    }
    Ok(())
}

/// The layers the check covers: 0, 1, 2, the last and every layer whose
/// stacks span more than one shard.
fn covered_layers(layers: &[Layer<'_>]) -> Vec<usize> {
    let last = layers.len().saturating_sub(1);
    layers
        .iter()
        .filter(|l| l.index < 3 || l.index == last || l.spans_shards())
        .map(|l| l.index)
        .collect()
}

/// The whole check (see the module doc); `Ok` only when every site passed.
fn check(bench: &Bench<'_>, n_host: usize, verbose: bool) -> Result<(), BenchError> {
    let (layers, xs) = (&bench.layers, &bench.xs);
    let covered = covered_layers(layers);
    let mut c = Checks {
        verbose,
        sites: 0,
        failed: Vec::new(),
        worst: 0.0,
    };
    for &li in &covered {
        let slots = distinct(&mut Rng::new(&[KEY_CHECK, li as u64]), n_host, WORKING_SET);
        check_layer(&mut c, &layers[li], &slots, &xs[li % N_X])?;
        let host = bench
            .hosts
            .get(li)
            .ok_or("no host layer to check the union shape")?;
        check_union(&mut c, &layers[li], host, bench.split, &slots, xs)?;
        if bench.r8.get(li).is_some_and(Option::is_some) {
            for shape in R8_CHECKS {
                check_union_r8(&mut c, bench, li, shape)?;
            }
        }
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

/// A timed arm's dispatch shape.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Shape {
    /// The host tier's: a gate+up group and a down group per layer.
    Engine,
    /// With `rows > 1`: one engine-shape call per row instead of one group
    /// over every row.
    EngineSep,
    /// One union call per layer over every row: each distinct expert read
    /// once, every column's weighted sum at the end.
    Union,
    /// The union call with gate and up through the row-lane Q3_K tile on
    /// the repacked working set.
    UnionR8,
    /// The `unionr8` call with the column tile on the file's rows: its
    /// dispatch, the union shape's kernel.
    UnionQ,
    /// One dispatch per expert matrix.
    PerMatrix,
    /// The engine shape's dispatches reading the file mapping, no kernel.
    ReadMmap,
    /// The same reads from the anonymous huge-page copy ([`ThpCopy`]).
    ReadThp,
}

impl Shape {
    const ALL: [Shape; 8] = [
        Shape::Engine,
        Shape::EngineSep,
        Shape::Union,
        Shape::UnionR8,
        Shape::UnionQ,
        Shape::PerMatrix,
        Shape::ReadMmap,
        Shape::ReadThp,
    ];

    fn name(self) -> &'static str {
        match self {
            Shape::Engine => "engine",
            Shape::EngineSep => "engine-sep",
            Shape::Union => "union",
            Shape::UnionR8 => "unionr8",
            Shape::UnionQ => "unionq",
            Shape::PerMatrix => "per-matrix",
            Shape::ReadMmap => "read-mmap",
            Shape::ReadThp => "read-thp",
        }
    }

    /// The shape runs one row per step only.
    fn one_row(self) -> bool {
        matches!(self, Shape::PerMatrix | Shape::ReadMmap | Shape::ReadThp)
    }

    /// One union call per layer: each distinct expert read once.
    fn is_union(self) -> bool {
        matches!(self, Shape::Union | Shape::UnionR8 | Shape::UnionQ)
    }
}

/// A timed arm: the dispatch shape and the host's share of a token's experts.
#[derive(Clone, Copy)]
struct Arm {
    shape: Shape,
    n_host: usize,
    /// Tokens per step (probe): each row reads its own activation column
    /// and `n_host` experts, shared with other rows only under `union`.
    rows: usize,
    /// The union ratio the arm asked for, `None` for disjoint rows.
    ratio: Option<f64>,
    /// Distinct experts per layer and token: `n_host * rows` for disjoint
    /// rows, else `round(ratio * n_host * rows)`.
    union: usize,
}

impl Arm {
    /// One arm of `--arms`; `union` is `--union`'s ratio, taken by a
    /// multi-row arm that names none of its own.
    fn parse(s: &str, union: Option<f64>) -> Result<Arm, String> {
        let (name, n) = s.split_once(':').ok_or_else(|| {
            format!(
                "arm {s:?}: want <engine|engine-sep|union|unionr8|unionq|per-matrix|read-mmap|read-thp>:<n_host>[x<rows>[u<r>]]"
            )
        })?;
        let shape = Shape::ALL
            .into_iter()
            .find(|sh| sh.name() == name)
            .ok_or_else(|| format!("arm {s:?}: unknown shape {name:?}"))?;
        let (n, rows, own) = match n.split_once('x') {
            Some((n, r)) => {
                let (r, own) = match r.split_once('u') {
                    Some((r, u)) => (
                        r,
                        Some(
                            u.parse::<f64>()
                                .map_err(|_| format!("arm {s:?}: union {u:?} is not a ratio"))?,
                        ),
                    ),
                    None => (r, None),
                };
                let rows = r
                    .parse::<usize>()
                    .map_err(|_| format!("arm {s:?}: rows {r:?} is not a count"))?;
                (n, rows, own)
            }
            None => (n, 1, None),
        };
        let n_host: usize = n
            .parse()
            .map_err(|_| format!("arm {s:?}: n_host {n:?} is not a count"))?;
        if n_host == 0 || rows == 0 {
            return Err(format!("arm {s:?}: n_host and rows must be positive"));
        }
        if shape.one_row() && rows > 1 {
            return Err(format!("arm {s:?}: the {name} shape runs one row"));
        }
        if shape.is_union() && (rows > UNION_MAX_COLS || n_host > EXPERTS_INTO_MAX) {
            return Err(format!(
                "arm {s:?}: a union call takes at most {UNION_MAX_COLS} rows of {EXPERTS_INTO_MAX} experts"
            ));
        }
        let slots = n_host * rows;
        let ratio = if rows > 1 { own.or(union) } else { None };
        let union = match ratio {
            None => slots,
            // `1/rows` written as a decimal lands a hair under it; the pool
            // still holds one row.
            Some(r) if r.is_finite() && r * rows as f64 >= 1.0 - 1e-6 && r <= 1.0 => {
                ((r * slots as f64).round() as usize).max(n_host)
            }
            Some(r) => {
                return Err(format!(
                    "arm {s:?}: union {r} is not in [1/{rows}, 1] — a row reads n_host distinct experts"
                ));
            }
        };
        if union > WORKING_SET {
            return Err(format!(
                "arm {s:?}: {union} distinct experts per layer, the working set holds {WORKING_SET}"
            ));
        }
        Ok(Arm {
            shape,
            n_host,
            rows,
            ratio,
            union,
        })
    }

    fn label(self) -> String {
        let shape = self.shape.name();
        match (self.rows, self.ratio) {
            (1, _) => format!("{shape}:{}", self.n_host),
            (rows, None) => format!("{shape}:{}x{rows}", self.n_host),
            (rows, Some(r)) => format!("{shape}:{}x{rows}u{r}", self.n_host),
        }
    }

    /// File bytes one token of this arm's dispatches stream: every row's
    /// slots, a shared expert once per row that reads it — except the union
    /// shapes, which stream each distinct expert once.
    fn bytes_per_token(self, layers: &[Layer<'_>]) -> u64 {
        if self.shape.is_union() {
            return self.union_bytes_per_token(layers);
        }
        layers
            .iter()
            .map(|l| (self.n_host * self.rows) as u64 * l.expert_bytes())
            .sum()
    }

    /// File bytes of one token's distinct experts: what a pass that read
    /// each once would read.
    fn union_bytes_per_token(self, layers: &[Layer<'_>]) -> u64 {
        layers
            .iter()
            .map(|l| self.union as u64 * l.expert_bytes())
            .sum()
    }

    /// Pool dispatches one token of this arm issues, by construction.
    fn dispatches_per_token(self, layers: &[Layer<'_>]) -> usize {
        match self.shape {
            Shape::PerMatrix => 3 * self.n_host * layers.len(),
            Shape::EngineSep => ENGINE_DISPATCHES * self.rows * layers.len(),
            Shape::Union | Shape::UnionR8 | Shape::UnionQ => {
                ENGINE_DISPATCHES * self.union.div_ceil(UNION_CHUNK) * layers.len()
            }
            Shape::Engine | Shape::ReadMmap | Shape::ReadThp => ENGINE_DISPATCHES * layers.len(),
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
            let mut union = None;
            while let Some(flag) = it.next() {
                let value = it.next().ok_or_else(|| format!("{flag} needs a value"))?;
                let bad = || format!("{flag} {value:?} is not a valid value");
                match flag {
                    "--rounds" => opts.rounds = value.parse().map_err(|_| bad())?,
                    "--seconds" => opts.seconds = value.parse().map_err(|_| bad())?,
                    "--warmup" => opts.warmup = value.parse().map_err(|_| bad())?,
                    "--arms" => arms = value.to_string(),
                    "--union" => union = Some(value.parse::<f64>().map_err(|_| bad())?),
                    other => return Err(format!("unknown option {other:?}")),
                }
            }
            let seconds_ok = opts.seconds.is_finite() && opts.seconds > 0.0;
            if opts.rounds == 0 || opts.warmup == 0 || !seconds_ok {
                return Err("--rounds, --warmup and --seconds must be positive".to_string());
            }
            opts.arms = arms
                .split(',')
                .map(|a| Arm::parse(a, union))
                .collect::<Result<_, _>>()?;
            Ok(Mode::Time(opts))
        }
        _ => Err("want --check or --time".to_string()),
    }
}

/// What every timed round reads: the layers, the activation columns, the
/// working set's pages, and the bytes the read shapes read with the sink
/// they fold into.
struct Bench<'a> {
    split: &'a Split,
    layers: Vec<Layer<'a>>,
    /// Every layer as the host tier's `HostLayer`, SwiGLU limit 0.
    hosts: Vec<HostLayer>,
    /// The `unionr8` shape's repacked gates and ups, per layer; `None` for a
    /// layer no arm and no check reads.
    r8: Vec<Option<R8Layer>>,
    xs: Vec<Tensor2>,
    pages: Pages<'a>,
    mapped: Vec<Vec<ExpertBytes<'a>>>,
    /// Present when an arm reads the huge-page copy.
    thp: Option<Vec<Vec<ExpertBytes<'a>>>>,
    sink: AtomicU64,
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
    /// The gate-and-up tile a `unionr8` or `unionq` arm runs on layer `l`.
    fn gate_up_tile(&self, shape: Shape, l: usize) -> Result<GateUpTile<'_>, BenchError> {
        match shape {
            Shape::UnionR8 => self
                .r8
                .get(l)
                .and_then(Option::as_ref)
                .map(|r8| GateUpTile::RowLane(r8))
                .ok_or_else(|| "unionr8 arm without the repacked working set".into()),
            _ => Ok(GateUpTile::Column),
        }
    }

    /// The working-set slots token `t` of `round` reads, per layer: `arm.rows`
    /// runs of `arm.n_host`, walking a pool of `arm.union` distinct slots in
    /// turn — the pool itself, in draw order, when the rows are disjoint.
    fn draw(&self, arm: Arm, round: usize, t: usize) -> Vec<Vec<usize>> {
        let slots = arm.n_host * arm.rows;
        (0..self.layers.len())
            .map(|l| {
                let key = [KEY_TOKEN, round as u64, t as u64, l as u64];
                let pool = distinct(&mut Rng::new(&key), arm.union, WORKING_SET);
                (0..slots).map(|i| pool[i % pool.len()]).collect()
            })
            .collect()
    }

    /// Every layer of one token. In layer `l`, `ids[l]` holds `rows` runs of
    /// `n_host` slots, row `i` reading activation column `(t + l + i) % N_X`.
    fn token(
        &self,
        arm: Arm,
        ids: &[Vec<usize>],
        t: usize,
        b: &mut Blocks,
        tally: &mut Tally,
    ) -> Result<(), BenchError> {
        for (l, layer) in self.layers.iter().enumerate() {
            let slots = &ids[l];
            let xs = || -> Vec<&Tensor2> {
                (0..arm.rows).map(|i| &self.xs[(t + l + i) % N_X]).collect()
            };
            match arm.shape {
                Shape::ReadMmap => read_layer(layer, &self.mapped[l], slots, &self.sink, tally),
                Shape::ReadThp => {
                    let set = self.thp.as_ref().ok_or("read-thp arm without a copy")?;
                    read_layer(layer, &set[l], slots, &self.sink, tally);
                }
                Shape::PerMatrix => {
                    per_matrix_layer(layer, slots, xs()[0], tally)?;
                }
                Shape::Engine | Shape::EngineSep if arm.rows == 1 => {
                    engine_layer(layer, slots, xs()[0], b, tally)?;
                }
                Shape::EngineSep => {
                    for (row, x) in slots.chunks_exact(arm.n_host).zip(&xs()) {
                        engine_layer(layer, row, x, b, tally)?;
                    }
                }
                Shape::Engine => engine_rows_layer(layer, slots, &xs(), b, tally)?,
                Shape::Union => {
                    let u = b.union.as_mut().ok_or("union arm without its blocks")?;
                    let host = self.hosts.get(l).ok_or("union arm without host layers")?;
                    let x = (t + l) % N_X;
                    union_layer(
                        host, self.split, layer, slots, arm.n_host, arm.union, u, x, tally,
                    )?;
                }
                Shape::UnionR8 | Shape::UnionQ => {
                    let u = b.r8.as_mut().ok_or("unionr8 arm without its blocks")?;
                    let host = self.hosts.get(l).ok_or("unionr8 arm without host layers")?;
                    let tile = self.gate_up_tile(arm.shape, l)?;
                    let x = (t + l) % N_X;
                    union_r8_layer(
                        host, self.split, layer, tile, slots, arm.n_host, arm.union, u, x, tally,
                    )?;
                }
            }
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
        let mut blocks = Blocks::new(&self.layers, arm.n_host * arm.rows)?;
        if arm.shape.is_union() {
            let embd = self.xs.first().ok_or("no activation columns")?.ne0;
            let ff = self.layers.first().ok_or("no layers")?.weight(0, GATE).n();
            if arm.shape == Shape::Union {
                blocks.union = Some(UnionBlocks::new(&self.xs, arm.rows, embd, ff)?);
            } else {
                // `unionr8` and `unionq` share the blocks.
                blocks.r8 = Some(R8Blocks::new(&self.xs, arm.rows, embd, ff, arm.union));
            }
        }
        let mut scratch = Tally::default();
        let mut warm_ms = Vec::with_capacity(warmup);
        for t in 0..warmup {
            let ids = self.draw(arm, round, t);
            let t0 = Instant::now();
            self.token(arm, &ids, t, &mut blocks, &mut scratch)?;
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
            let ids = self.draw(arm, round, t);
            let t0 = Instant::now();
            self.token(arm, &ids, t, &mut blocks, &mut tally)?;
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
            let union_bytes = arm.union_bytes_per_token(&bench.layers);
            let (min, mean, max) = spread(&r.ms);
            let ok =
                lease && r.resident_before == total && r.resident_after == total && r.majflt == 0;
            println!(
                "time threads={threads} round={}/{} arm={} tokens={} warmup={} ms_min={min:.3} ms_mean={mean:.3} ms_max={max:.3} \
                 host_bytes_per_token={bytes} union_per_layer={} union_bytes_per_token={union_bytes} \
                 gbps_mean={:.2} gbps_best={:.2} dispatches_per_token={:.2} expected={} \
                 minflt={} majflt={} resident_before={}/{total} resident_after={}/{total} lease={} admissible={}",
                round + 1,
                opts.rounds,
                arm.label(),
                r.ms.len(),
                opts.warmup,
                arm.union,
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
        let thp = if arm.shape == Shape::ReadThp {
            format!(
                " thp_enabled={} anon_huge_kb={}",
                thp_mode(),
                anon_huge_kb()?
            )
        } else {
            String::new()
        };
        println!(
            "summary threads={threads} arm={} rounds={} tokens={} ms_min={min:.3} ms_mean={mean:.3} round_means=[{}] \
             gbps_mean={:.2} gbps_best={:.2} admissible={}{thp}",
            arm.label(),
            rs.len(),
            all.len(),
            means.join(","),
            bytes as f64 / mean / 1e6,
            bytes as f64 / min / 1e6,
            if ok { "yes" } else { "no" }
        );
    }
    if opts
        .arms
        .iter()
        .any(|a| matches!(a.shape, Shape::ReadMmap | Shape::ReadThp))
    {
        println!(
            "v41host read_sink={:#018x}",
            bench.sink.load(Ordering::Relaxed)
        );
    }
    Ok(())
}

/// Every layer as a `HostLayer` over the same stacks, SwiGLU limit 0 (the
/// plain combine the other shapes run).
fn host_layers(
    split: &Split,
    meta: &Meta,
    layers: &[Layer<'_>],
) -> Result<Vec<HostLayer>, BenchError> {
    layers
        .iter()
        .map(|l| {
            let spec = HostLayerSpec {
                gate: &l.stacks[GATE].info.name,
                up: &l.stacks[UP].info.name,
                down: &l.stacks[DOWN].info.name,
                n_expert: meta.n_expert,
                embd: meta.embd,
                ff: meta.ff,
                swiglu_limit: 0.0,
            };
            Ok(HostLayer::build(split, &spec)?)
        })
        .collect()
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
    let mapped = mapped_bytes(&layers);
    // The copy is made before the working set is paged in: filling it can
    // evict page-cache pages, and the populate below brings them back.
    let wants_thp =
        matches!(&mode, Mode::Time(o) if o.arms.iter().any(|a| a.shape == Shape::ReadThp));
    let copy = if wants_thp {
        Some(ThpCopy::make(&mapped)?)
    } else {
        None
    };
    // Repacked before the working set is paged in, as the copy above: the
    // anonymous memory it fills can evict page-cache pages.
    let r8_layers = match &mode {
        Mode::Check => covered_layers(&layers),
        Mode::Time(o) if o.arms.iter().any(|a| a.shape == Shape::UnionR8) => {
            (0..layers.len()).collect()
        }
        Mode::Time(_) => Vec::new(),
    };
    let t0 = Instant::now();
    let r8 = repack_r8(&layers, &r8_layers)?;
    if !r8_layers.is_empty() {
        let bytes: usize = r8
            .iter()
            .flatten()
            .flat_map(|l| l.iter().flatten())
            .map(Vec::len)
            .sum();
        println!(
            "v41host unionr8 repacked layers={} bytes={bytes} repack_s={:.2}",
            r8_layers.len(),
            t0.elapsed().as_secs_f64()
        );
    }
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
    let hosts = host_layers(&split, &meta, &layers)?;
    let bench = Bench {
        split: &split,
        layers,
        hosts,
        r8,
        xs: activations(meta.embd),
        pages,
        mapped,
        thp: copy.as_ref().map(ThpCopy::bytes),
        sink: AtomicU64::new(0),
    };
    match mode {
        Mode::Check => {
            check(&bench, meta.n_used, true)?;
            let arms = R8_CHECK_ARMS
                .split(',')
                .map(|a| Arm::parse(a, None))
                .collect::<Result<Vec<_>, _>>()?;
            check_r8_arms(&bench, &arms, &covered_layers(&bench.layers))
        }
        Mode::Time(opts) => {
            let lease = std::env::var("BLOOMERY_HOST_LEASE").is_ok_and(|v| v == "1");
            if !lease {
                println!(
                    "[not under lease] — run through tools/ref/host-rate.sh; these numbers are not admissible"
                );
            }
            check(&bench, meta.n_used, false)?;
            let every: Vec<usize> = (0..bench.layers.len()).collect();
            check_r8_arms(&bench, &opts.arms, &every)?;
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
