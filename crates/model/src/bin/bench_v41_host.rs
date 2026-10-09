//! Bench for the host leg of a DeepSeek-V4.1 decode token, and of the
//! Qwen3.8-Flash-Next (`qwen4exp`) routed experts the host union serves.
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
//! copied — file-backed page-cache pages, as serving reads them. The file's
//! `general.architecture` picks the family ([`Family`]). A `qwen4exp` file
//! reads its hyperparameters, its routed stacks' names and its host layers
//! from `arch::qwen35moe` (`Hparams`, `names`, `host::layers`); any other
//! file's routed stacks are its three-dimensional tensors whose last dim is
//! the file's `expert_count`: down by its per-expert shape `[ff, embd]`, gate
//! and up (both `[embd, ff]`) by the word their names carry. Each layer's working
//! set is [`WORKING_SET`] experts drawn once by a seeded generator, and a
//! token draws its `n_host` experts from that set, distinct within the
//! layer. The working set is paged in before anything runs
//! (`madvise(MADV_WILLNEED)`, then one read per page). Around every timed
//! arm its page-cache residency (`mincore`) and the process's page-fault
//! counters are recorded: a page read from NVMe makes the number a disk
//! measurement, and such an arm prints `admissible=no`.
//!
//! `--check [--arms A,B,...]`: for layers 0, 1, 2, the last layer and every layer whose
//! stacks span more than one shard — and, for a `qwen4exp` file, the first
//! layer of every routed type triple (gate, up, down) those lack — one token
//! and every one of its `expert_used_count` experts. Six output rows of gate,
//! up and down are compared with an f64 reference over the same bytes:
//! `gguf::dequant_row` for the weight rows, and for the activation the bytes
//! `qdot::quantize_col` makes of the kernel's own input (block_q8_K for q3_K,
//! block_q8_2_x4 for q4_K and q5_K, block_q8_2_x4 then block_q8_2 tails for
//! q5_1 and q8_0), decoded exactly — the float sum order is the only
//! difference. A q5_1 kernel adds each block's min times the activation
//! block's stored sum; the reference adds that term from the stored sum too.
//! The band is [`BAND`]. The SwiGLU combine the down dispatch
//! produced must equal `qdot::swiglu` over the gate and up outputs bit for
//! bit and sit within the band of an f64 SiLU, and the per-matrix shape must
//! reproduce the engine shape's outputs bit for bit. A miss names the layer,
//! the expert and the matrix, and the run exits non-zero. On the same layers
//! the union shape runs two rows — the checked token's experts, and the same
//! set shifted by one with one expert of its own — and each row's output
//! must equal, bit for bit, the list-order sum of the engine shape's downs
//! for that row at the union shape's weights. Where the working set was
//! repacked for the `unionr8` shape (every checked layer under `--check`,
//! every layer when an arm is `unionr8`), the checked layers also run one
//! token of each of [`R8_CHECKS`] through the `union`, `union5`, `unionr8` and
//! `unionq` shapes, whose outputs must be equal bit for bit, and `--check`
//! runs the pre-timing arm check below for the arms of [`R8_CHECK_ARMS`] on
//! the checked layers — or, given `--arms`, for those arms on every layer,
//! the check a `--time` run makes before it times them, without the timing.
//! The `unionr8` and `unionq` shapes take Q3_K gates and ups, which a
//! `qwen4exp` file has none of: its `--check` runs neither and says so, and
//! its default arm check is [`Q38_CHECK_ARMS`].
//!
//! `--time` (lead-only, under `tools/ref/host-rate.sh`, which owns the lease,
//! the witnesses and the thread sweep): the check first, refusing to time if
//! it fails; then `--rounds` rounds of every arm, the arm order rotated each
//! round. Per arm and round, `--warmup` untimed tokens, then as many timed
//! tokens as fill the arm's share of `--seconds` at the first warm-up's pace;
//! later rounds keep that count. Every dispatch is timed on its own too, by
//! kind and weight bytes: bytes per dispatch is the variable this bench is
//! for. A union arm's lines also carry its columns per distinct expert
//! (`cols_per_expert`) and the time per distinct expert of one layer call
//! (`us_per_expert`: a dispatch line's mean call over the arm's distinct
//! experts; a summary line's mean token over its layers and distinct experts).
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
//! The `union5` shape is that pass: per layer one
//! `moe::HostLayer::experts_union_into` over the `rows` columns and their
//! lists (each row's `n_host` working-set experts at weight `1 / n_host`),
//! which reads each distinct expert once and ends in every column's weighted
//! sum, in five pool dispatches a layer (two at or under
//! `ops::DEFER_MAX_COLS` rows). Its layers are built with a SwiGLU limit of
//! 0, which clamps nothing, so its combine is the plain one the other shapes
//! run. The call is timed whole, as one `union5` dispatch entry whose bytes
//! are the layer's distinct experts.
//!
//! The `unioncard` shape is the `union5` pass over qdot's card-rule kernels
//! — the card engine's own rule (q8_1 per 128 values, the Q3_K/Q4_K lane
//! walks and the warp tree, the SwiGLU through `v_expf`) on the host, so a
//! routed expert's contribution holds the same bits whichever tier computes
//! it: x quantized once per column, each distinct expert's gate·up and down
//! as m-column walks, the h quantizer, the plain combine. Its `--check`
//! holds every row's output equal to the one-column card-rule calls, bit
//! for bit; E1 times it beside `union5` at m = 1, 8 and 16.
//!
//! The `union` shape is the same pass in the chunked flow the host tier ran
//! before (see [`union_chunks_call`]): the distinct experts in chunks of
//! [`UNION_CHUNK`], a gate/up and a down group dispatch per chunk
//! (`ops::matmul_q_group_cols_into`), each chunk's downs but the last's
//! copied aside, then the sums — at least `2 · ceil(union / UNION_CHUNK)`
//! pool dispatches a layer. `union5` against `union` is the dispatch flow;
//! their outputs are equal bit for bit.
//!
//! The `unionr8` shape is the same call with gate and up through the
//! row-lane Q3_K tile (`qdot::dot_q3k_r8_cols`): the working set's gates and
//! ups are repacked once at start-up (`qdot::repack_q3k_r8`, into anonymous
//! memory, before the working set is paged in), and per layer the `union`
//! shape's plan, chunks and down dispatch run unchanged. Its gate and up
//! dispatch hands out (expert, 8-row group) units off one counter, `b<B>` on
//! the arm (default 1) at a time, and a call of at most `ops::DEFER_MAX_COLS`
//! columns quantizes `x` as claims inside that dispatch; a wider one
//! quantizes it over the pool first, as the `union` shape does. Each
//! distinct expert keeps its own down block, so no chunk copies its downs
//! aside. The `unionq` shape is that call with the column tile on the file's
//! rows (`qdot::dot_row_cols` row by row, as the union call's row dispatch
//! runs it): `unionr8` against `unionq` is the kernel alone, `unionq` against
//! `union` the gate/up dispatch and the narrow call's quantization. Before
//! timing, one token of every `union5`, `unionr8` and `unionq` arm runs
//! through its shape and the `union` shape on every layer, and their outputs
//! must be equal bit for bit.

use std::ffi::{c_int, c_void};
use std::ops::Range;
use std::process::ExitCode;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

use gguf::quant::half_to_f32;
use gguf::{GgmlType, Gguf, Split, TensorInfo, dequant_row};
use model::ModelError;
use model::arch::qwen35moe::hparams::Hparams;
use model::arch::qwen35moe::{host as qwen4exp_host, names as qwen4exp_names};
use model::moe::{HostLayer, HostLayerSpec, UNION_MAX_COLS, UnionScratch, expert_view};
use model::ops::{
    DEFER_MAX_COLS, GroupInput, Lanes, QuantizedCols, ShardTensor, Tensor2, WarmSpan, Weight,
    matmul_q, matmul_q_group_cols_into, matmul_q_group_into_on, matmul_q_group_swiglu,
    matmul_q_group_swiglu_into_on, warm_share,
};
use model::r8file::R8Source;
use threads::CcdMap;

type BenchError = Box<dyn std::error::Error>;

/// Experts per layer in the working set: forty layers of them are far above
/// the L3 and far below free RAM, and a union arm's 32 distinct experts fit.
const WORKING_SET: usize = 32;
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
/// Distinct experts per chunk of the chunked union flow (the `union`,
/// `unionr8` and `unionq` shapes), a geometry of the flow and no model's
/// width: a chunk's gate and up fill the group bookkeeping's inline block
/// (16 pairs) and its downs fit the claim table.
const UNION_CHUNK: usize = 8;
/// The arms `--check` runs the pre-timing arm check for, on the checked
/// layers: the union sitting's shapes.
const R8_CHECK_ARMS: &str = "union5:4x8u0.125,unionr8:4x8u0.125,unionq:4x8u0.125,\
                             union5:4x16u0.0625,unionr8:4x16u0.0625,unionq:4x16u0.0625";
/// The arms a `qwen4exp` file's `--check` runs the pre-timing arm check for:
/// [`R8_CHECK_ARMS`]' `union5` arms (its gates and ups are not Q3_K).
const Q38_CHECK_ARMS: &str = "union5:4x8u0.125,union5:4x16u0.0625";
/// The `unioncard` arms a plain `--check` runs on the checked layers: the
/// card-rule pass at one, eight and sixteen columns an expert — the `m`
/// points the host-union row's `t(m) = max(W, a + c·m)` was fit from.
const CARD_CHECK_ARMS: &str = "unioncard:4,unioncard:4x8u0.125,unioncard:4x16u0.0625";
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
/// The refusal of a warm shape, or of a plain `--check` (it runs the warm
/// arms), on a pool with no CCD map.
const NO_CCD_MAP: &str = "a warm arm needs the pool's CCD map: a worker's pin failed or the caller \
                          is not pinned (BLOOMERY_PIN_MAIN=0)";
/// The warm arms a plain `--check` runs on the checked layers: a warm of two
/// experts of the engine shape's three, and of two of the union call's six.
const WARM_CHECK_ARMS: &str = "engine-warm:3w2,union5-warm:3x4u0.5w2";

const USAGE: &str = "usage: bench_v41_host --check [--arms A,B,...]
       bench_v41_host --time [--rounds N] [--seconds S] [--warmup W] [--arms A,B,...]
       bench_v41_host --time ... [--union R]
  an arm is <engine|engine-sep|union|union5|unionr8|unionq|unioncard|per-matrix|read-mmap|read-thp>:<n_host>[x<rows>[u<r>]][b<B>]; the default is engine:6,engine:5,engine:3,per-matrix:6
  union: one union call per layer over the rows in the chunked flow (rows <= 512, n_host <= 8)
  union5: the engine's union call, five pool dispatches a layer (same bounds)
  unionr8: the union call with gate and up through the row-lane Q3_K tile (same bounds)
  unionq: the unionr8 call with the union call's column tile (its control: same dispatch)
  unioncard: the union5 pass over qdot's card-rule kernels (the card's own bits; same bounds)
  u<r> (or --union R for every multi-row arm without its own): the rows share experts, the layer's distinct experts
  are round(r x n_host x rows), 1/rows <= r <= 1
  b<B> (unionr8 and unionq only): the gate/up dispatch hands out B units a claim, default 1
  --check --arms: the pre-timing check of those arms on every layer, no timing
  the model is $BLOOMERY_REF_MODEL (tools/box.sh exports it from the BLOOMERY_MODEL profile: deepseek41 or qwen4exp)";

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
    /// The values `family` reads: a `qwen4exp` file's from its [`Hparams`],
    /// any other's from the five keys.
    fn read(split: &Split, family: &Family) -> Result<Meta, BenchError> {
        if let Family::Qwen4exp(hp) = family {
            return Meta {
                blocks: hp.n_layer,
                embd: hp.n_embd,
                ff: hp.expert_ff,
                n_expert: hp.n_expert,
                n_used: hp.n_used,
            }
            .fits();
        }
        let get = |suffix: &str| -> Result<usize, BenchError> {
            let v = split
                .arch_get_u64(suffix)
                .ok_or_else(|| format!("metadata key {} is absent", split.arch_key(suffix)))?;
            Ok(usize::try_from(v)?)
        };
        Meta {
            blocks: get("block_count")?,
            embd: get("embedding_length")?,
            ff: get("expert_feed_forward_length")?,
            n_expert: get("expert_count")?,
            n_used: get("expert_used_count")?,
        }
        .fits()
    }

    /// `self` when the working set holds a token's experts, else the named error.
    fn fits(self) -> Result<Meta, BenchError> {
        if self.n_expert < WORKING_SET || self.n_used == 0 || self.n_used > WORKING_SET {
            return Err(format!(
                "{} experts, {} used per token: a working set of {WORKING_SET} does not fit",
                self.n_expert, self.n_used
            )
            .into());
        }
        Ok(self)
    }
}

/// How the bench finds a file's routed stacks and builds its host layers,
/// picked by the file's `general.architecture`.
enum Family {
    /// Any architecture but `qwen4exp` (DeepSeek-V4.1): the stacks found by
    /// their shapes ([`find_stacks`]), the host layers built from them.
    Discovered,
    /// Qwen3.8-Flash-Next: the hyperparameters, stack names and host layers
    /// `arch::qwen35moe` reads and builds.
    Qwen4exp(Box<Hparams>),
}

impl Family {
    fn of(split: &Split) -> Result<Family, BenchError> {
        match split.architecture() {
            Some("qwen4exp") => Ok(Family::Qwen4exp(Box::new(Hparams::read(split)?))),
            _ => Ok(Family::Discovered),
        }
    }

    fn name(&self) -> &'static str {
        match self {
            Family::Discovered => "discovered",
            Family::Qwen4exp(_) => "qwen4exp",
        }
    }

    /// Every layer's routed stacks, `[gate, up, down]`.
    fn stacks<'a>(&self, split: &'a Split, meta: &Meta) -> Result<Vec<[Stack<'a>; 3]>, BenchError> {
        match self {
            Family::Discovered => find_stacks(split, meta),
            Family::Qwen4exp(_) => (0..meta.blocks)
                .map(|l| {
                    let stack = |m: usize, name: String| -> Result<Stack<'a>, BenchError> {
                        let (shard, info) = split
                            .find(&name)
                            .ok_or_else(|| format!("{name}: not in the file"))?;
                        let rest = layer_part(&name).map_or("", |(_, rest)| rest);
                        if info.dims.len() != 3 || info.dims[2] != meta.n_expert as u64 {
                            return Err(format!(
                                "{name}: dims {:?}, not a stack of {} experts",
                                info.dims, meta.n_expert
                            )
                            .into());
                        }
                        if matrix_of(info, rest, meta)? != m {
                            return Err(format!(
                                "{name}: not the {} stack its name says",
                                MATRIX[m]
                            )
                            .into());
                        }
                        Ok(Stack { shard, info })
                    };
                    Ok([
                        stack(GATE, qwen4exp_names::ffn_gate_exps(l))?,
                        stack(UP, qwen4exp_names::ffn_up_exps(l))?,
                        stack(DOWN, qwen4exp_names::ffn_down_exps(l))?,
                    ])
                })
                .collect(),
        }
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
fn build_layers<'a>(
    split: &'a Split,
    meta: &Meta,
    family: &Family,
) -> Result<Vec<Layer<'a>>, BenchError> {
    let mut layers = Vec::with_capacity(meta.blocks);
    for (index, stacks) in family.stacks(split, meta)?.into_iter().enumerate() {
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
    /// The `union` shape's, `None` for the others.
    chunks: Option<ChunkBlocks>,
    /// The `union5` shape's, `None` for the others.
    union: Option<UnionBlocks>,
    /// The `unionr8` shape's, `None` for the others.
    r8: Option<R8Blocks>,
    /// The `unioncard` shape's, `None` for the others.
    card: Option<CardBlocks>,
}

/// The rows' activation blocks of a union shape: block `o` holds the `rows`
/// columns `(o + i) % N_X` a token reads at offset `o`.
fn union_xs(xs: &[Tensor2], rows: usize, embd: usize) -> Vec<Tensor2> {
    (0..N_X)
        .map(|o| {
            let mut data = Vec::with_capacity(embd * rows);
            for i in 0..rows {
                data.extend_from_slice(&xs[(o + i) % N_X].data);
            }
            Tensor2::from_vec(embd, rows, data)
        })
        .collect()
}

/// The `union5` shape's blocks, made before the round's first token: the
/// rows' activation blocks ([`union_xs`]), then the call's scratch (made for
/// `rows` columns), its `embd × rows` output and the rows' list entries.
struct UnionBlocks {
    xs: Vec<Tensor2>,
    scratch: UnionScratch,
    out: Vec<f32>,
    lists: Vec<Vec<(u32, f32)>>,
}

impl UnionBlocks {
    fn new(
        xs: &[Tensor2],
        rows: usize,
        embd: usize,
        ff: usize,
        n_used: usize,
    ) -> Result<UnionBlocks, ModelError> {
        Ok(UnionBlocks {
            xs: union_xs(xs, rows, embd),
            scratch: UnionScratch::new_routed(embd, ff, rows, n_used)?,
            out: vec![0.0; embd * rows],
            lists: vec![vec![(0, 0.0); n_used]; rows],
        })
    }
}

/// A raw pointer the pool's Sync closures can carry: each dispatch's chunks
/// partition the buffer's index space, one owner an index (the ops.rs pool
/// pattern), and the join publishes the writes.
struct ChunkPtr<T>(*mut T);
// SAFETY: the pointer is shared between a dispatch's closures only, which
// the pool runs over disjoint index chunks; the join before the next read
// publishes every write.
unsafe impl<T> Send for ChunkPtr<T> {}
// SAFETY: as `Send` — no closure reads an index another chunk writes.
unsafe impl<T> Sync for ChunkPtr<T> {}
impl<T: Copy> ChunkPtr<T> {
    fn write(&self, i: usize, v: T) {
        // SAFETY: this closure's pool chunk owns index `i` — the chunks
        // partition the index space and the join publishes the writes.
        unsafe {
            *self.0.add(i) = v;
        }
    }

    fn read(&self, i: usize) -> T {
        // SAFETY: as `write` — the index is this chunk's, and the joins
        // before this dispatch published every write it reads.
        unsafe { *self.0.add(i) }
    }
}

impl<T> ChunkPtr<T> {
    /// `f` on element `i`, which this closure's pool chunk owns.
    fn with_mut<R>(&self, i: usize, f: impl FnOnce(&mut T) -> R) -> R {
        // SAFETY: as `write` — the chunks partition the index space, so no
        // other closure touches element `i` while `f` holds it.
        f(unsafe { &mut *self.0.add(i) })
    }
}

/// The `unioncard` shape's blocks, made before the round's first token: the
/// rows' activation blocks and the pass's buffers — x's q8_1 form (one a
/// column, refilled every call), each slot's h and its q8_1 form, the
/// downs, the output. The pass itself owns no allocation.
struct CardBlocks {
    xs: Vec<Tensor2>,
    xcols: Vec<qdot::CardQ81>,
    hbuf: Vec<f32>,
    hcols: Vec<qdot::CardQ81>,
    downbuf: Vec<f32>,
    out: Vec<f32>,
    /// Whether the pass has said it skips a layer yet (its down is not
    /// q4_K — the card rule covers q4_K downs only, this round).
    skip_said: bool,
}

impl CardBlocks {
    fn new(xs: &[Tensor2], rows: usize, embd: usize, ff: usize, n_used: usize) -> CardBlocks {
        let slots = rows * n_used;
        CardBlocks {
            xs: union_xs(xs, rows, embd),
            xcols: (0..rows).map(|_| qdot::CardQ81::for_k(embd)).collect(),
            hbuf: vec![0.0; slots * ff],
            hcols: (0..slots).map(|_| qdot::CardQ81::for_k(ff)).collect(),
            downbuf: vec![0.0; slots * embd],
            out: vec![0.0; embd * rows],
            skip_said: false,
        }
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
            chunks: None,
            union: None,
            r8: None,
            card: None,
        })
    }
}

/// Row `i` of `lists` listing its `n_host` slots of `slots` at weight
/// `1 / n_host`, as every union shape lists them.
fn fill_lists(
    lists: &mut [Vec<(u32, f32)>],
    l: &Layer<'_>,
    slots: &[usize],
    n_host: usize,
) -> Result<(), BenchError> {
    let w = 1.0 / n_host as f32;
    for (row, run) in lists.iter_mut().zip(slots.chunks_exact(n_host)) {
        for (entry, &s) in row.iter_mut().zip(run) {
            let id = u32::try_from(l.experts[s].id)
                .map_err(|_| format!("expert {} past u32", l.experts[s].id))?;
            *entry = (id, w);
        }
    }
    Ok(())
}

/// The `union5` shape: one `HostLayer::experts_union_into` over the token's
/// rows, row `i` listing its `n_host` slots' experts at weight `1 / n_host`.
/// Tallied whole, under the bytes of the layer's distinct experts.
#[allow(clippy::too_many_arguments)] // one call's whole input: the layer three ways, its lists' parts, the blocks and the tally
fn union5_layer(
    host: &HostLayer,
    split: &Split,
    l: &Layer<'_>,
    slots: &[usize],
    n_host: usize,
    distinct: usize,
    u: &mut UnionBlocks,
    x: usize,
    tally: &mut Tally,
) -> Result<(), BenchError> {
    fill_lists(&mut u.lists, l, slots, n_host)?;
    let lists: Vec<&[(u32, f32)]> = u.lists.iter().map(|r| &r[..n_host]).collect();
    let t0 = Instant::now();
    host.experts_union_into(
        R8Source::rows(split),
        &u.xs[x],
        &lists,
        &mut u.out,
        &mut u.scratch,
    )?;
    tally.add("union5", distinct as u64 * l.expert_bytes(), t0);
    Ok(())
}

/// The `unioncard` shape: the `union5` pass over qdot's card-rule kernels —
/// the card engine's own bits, on the host. Five pool dispatches a layer:
/// x quantized once per column (q8_1 per 128 values), every distinct
/// expert's gate·up rows over the columns that list it (one m-column card
/// walk a row, SwiGLU inline), each slot's h quantized, every distinct
/// expert's down rows over its slots' h columns, then each row's weighted
/// sum in its list order from zero — the union call's combine. Each
/// column's every value is the one-column kernels' bit for bit. Tallied
/// whole, under the bytes of the layer's distinct experts.
fn unioncard_layer(
    l: &Layer<'_>,
    slots: &[usize],
    n_host: usize,
    c: &mut CardBlocks,
    x: usize,
    tally: &mut Tally,
) -> Result<(), BenchError> {
    let rows = slots.len() / n_host;
    let (embd, ff) = (c.xs[0].ne0, c.hbuf.len() / (rows * n_host));
    if l.stacks[GATE].info.ty != GgmlType::Q3_K || l.stacks[UP].info.ty != GgmlType::Q3_K {
        return Err(format!(
            "the unioncard shape takes q3_K gates and ups; layer {}'s are {:?} and {:?}",
            l.index, l.stacks[GATE].info.ty, l.stacks[UP].info.ty
        )
        .into());
    }
    if l.stacks[DOWN].info.ty != GgmlType::Q4_K {
        // The card rule this round covers the q4_K down; a q5_K down layer
        // (V4.1's first two) is the host tier's in every plan and has no
        // card path to mirror. Said once; the pass skips the layer.
        if !c.skip_said {
            println!(
                "unioncard: layers whose down is not q4_K (layer {} is {}) are skipped — no card rule this round",
                l.index, l.stacks[DOWN].info.ty
            );
            c.skip_said = true;
        }
        return Ok(());
    }
    // The pass's plan: the distinct experts in first-appearance order, each
    // with its slots (row, list position).
    let mut plan: Vec<(usize, Vec<(usize, usize)>)> = Vec::new();
    for row in 0..rows {
        for j in 0..n_host {
            let s = slots[row * n_host + j];
            match plan.iter_mut().find(|(e, _)| *e == s) {
                Some((_, es)) => es.push((row, j)),
                None => plan.push((s, vec![(row, j)])),
            }
        }
    }
    let w = 1.0 / n_host as f32;
    let m = plan.iter().map(|(_, es)| es.len()).max().unwrap_or(1);
    let t0 = Instant::now();
    // x quantized once per column, each form its chunk's own.
    let CardBlocks {
        xs,
        xcols,
        hbuf,
        hcols,
        downbuf,
        out,
        skip_said: _,
    } = c;
    {
        let forms = ChunkPtr(xcols.as_mut_ptr());
        let x = &xs[x];
        threads::pool().for_each_chunk(rows, |range| {
            for r in range {
                forms.with_mut(r, |f| {
                    qdot::card_q8_1_fill(&x.data[r * embd..(r + 1) * embd], f);
                });
            }
        });
    }
    // Gate·up·SwiGLU: one dispatch over every distinct expert's rows, each
    // row's m-column walk over the x forms of the rows that list it. The
    // SwiGLU limit is 0, the union shapes' own (their layers clamp nothing).
    {
        let xcols = &*xcols;
        let hbuf = ChunkPtr(hbuf.as_mut_ptr());
        /// One distinct expert's walk inputs: its gate and up stacks and the
        /// slots (row, list position) that list it.
        type ExpertWalk<'a> = (&'a [u8], &'a [u8], &'a [(usize, usize)]);
        let gates: Vec<ExpertWalk<'_>> = plan
            .iter()
            .map(|(s, es)| {
                let g = l.weight(*s, GATE);
                let u = l.weight(*s, UP);

                (g.bytes(), u.bytes(), es.as_slice())
            })
            .collect();
        let rb = gates.first().ok_or("unioncard: no experts")?.0.len() / ff;
        threads::pool().for_each_chunk(plan.len() * ff, |range| {
            let (mut gu, mut up) = (vec![0.0f32; m], vec![0.0f32; m]);
            let mut cols: Vec<&qdot::CardQ81> = Vec::with_capacity(m);
            for r in range {
                let (gb, ub, es) = &gates[r / ff];
                cols.clear();
                for &(row, _) in es.iter() {
                    cols.push(&xcols[row]);
                }
                qdot::card_q3k_dot_row_cols(&gb[r % ff * rb..][..rb], &cols, &mut gu[..es.len()])
                    .unwrap();
                qdot::card_q3k_dot_row_cols(&ub[r % ff * rb..][..rb], &cols, &mut up[..es.len()])
                    .unwrap();
                for (j, &(row, lj)) in es.iter().enumerate() {
                    hbuf.write(
                        (row * n_host + lj) * ff + r % ff,
                        qdot::card_swiglu_clamp_1(gu[j], up[j], 0.0),
                    );
                }
            }
        });
    }
    // Each slot's h quantized, each form its chunk's own.
    {
        let hbuf = &*hbuf;
        let forms = ChunkPtr(hcols.as_mut_ptr());
        threads::pool().for_each_chunk(rows * n_host, |range| {
            for s in range {
                forms.with_mut(s, |f| qdot::card_q8_1_fill(&hbuf[s * ff..(s + 1) * ff], f));
            }
        });
    }
    // The downs: one dispatch over every distinct expert's down rows.
    {
        let hcols = &*hcols;
        let downbuf = ChunkPtr(downbuf.as_mut_ptr());
        /// One distinct expert's down inputs: its stack and its slots.
        type ExpertDown<'a> = (&'a [u8], &'a [(usize, usize)]);
        let downs: Vec<ExpertDown<'_>> = plan
            .iter()
            .map(|(s, es)| (l.weight(*s, DOWN).bytes(), es.as_slice()))
            .collect();
        let rb = downs.first().ok_or("unioncard: no experts")?.0.len() / embd;
        threads::pool().for_each_chunk(plan.len() * embd, |range| {
            let mut dv = vec![0.0f32; m];
            let mut cols: Vec<&qdot::CardQ81> = Vec::with_capacity(m);
            for r in range {
                let (db, es) = &downs[r / embd];
                cols.clear();
                for &(row, lj) in es.iter() {
                    cols.push(&hcols[row * n_host + lj]);
                }
                qdot::card_q4k_dot_row_cols(&db[r % embd * rb..][..rb], &cols, &mut dv[..es.len()])
                    .unwrap();
                for (j, &(row, lj)) in es.iter().enumerate() {
                    downbuf.write((row * n_host + lj) * embd + r % embd, dv[j]);
                }
            }
        });
    }
    // Each row's weighted sum, its list order from zero (the union call's
    // combine: a multiply and an add, in that order).
    {
        let (downbuf, out) = (&*downbuf, ChunkPtr(out.as_mut_ptr()));
        threads::pool().for_each_chunk(rows, |range| {
            for row in range {
                for i in 0..embd {
                    out.write(row * embd + i, 0.0);
                }
                for j in 0..n_host {
                    let dv = &downbuf[(row * n_host + j) * embd..(row * n_host + j + 1) * embd];
                    for (i, &v) in dv.iter().enumerate() {
                        let at = row * embd + i;
                        out.write(at, out.read(at) + w * v);
                    }
                }
            }
        });
    }
    tally.add("unioncard", plan.len() as u64 * l.expert_bytes(), t0);
    Ok(())
}

/// The `union` shape: [`union_chunks_call`] over the token's rows, row `i`
/// listing its `n_host` slots' experts at weight `1 / n_host`. Tallied whole,
/// under the bytes of the layer's distinct experts.
#[allow(clippy::too_many_arguments)] // one call's whole input: the layer three ways, its lists' parts, the blocks and the tally
fn union_layer(
    host: &HostLayer,
    split: &Split,
    l: &Layer<'_>,
    slots: &[usize],
    n_host: usize,
    distinct: usize,
    u: &mut ChunkBlocks,
    x: usize,
    tally: &mut Tally,
) -> Result<(), BenchError> {
    fill_lists(&mut u.lists, l, slots, n_host)?;
    let t0 = Instant::now();
    union_chunks_call(host, split, l, x, n_host, u)?;
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
// among the participants; a `T` written on one thread and read on another
// must be `Send`.
unsafe impl<T: Send> Send for Cells<T> {}
// SAFETY: as for `Send` — no cell is written by two participants.
unsafe impl<T: Send> Sync for Cells<T> {}

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
    /// Room for `cols` columns of `n_used` experts each.
    fn with_room(cols: usize, n_used: usize) -> R8Plan {
        let slots = cols * n_used;
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
        lists: &[Vec<(u32, f32)>],
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
/// the output and the rows' lists, and the units the gate/up dispatch hands
/// out a claim (the arm's `b<B>`).
struct R8Blocks {
    xs: Vec<Tensor2>,
    xq: Vec<u8>,
    gate_up: [Tensor2; 2 * UNION_CHUNK],
    pars: [Tensor2; UNION_CHUNK],
    downs: Vec<Tensor2>,
    out: Vec<f32>,
    plan: R8Plan,
    lists: Vec<Vec<(u32, f32)>>,
    claim_block: usize,
}

impl R8Blocks {
    /// Blocks for calls over `rows` columns of up to `n_used` experts and at
    /// most `distinct` experts, `claim_block` units a claim.
    fn new(
        xs: &[Tensor2],
        (rows, embd, ff): (usize, usize, usize),
        n_used: usize,
        distinct: usize,
        claim_block: usize,
    ) -> R8Blocks {
        R8Blocks {
            claim_block,
            xs: union_xs(xs, rows, embd),
            xq: vec![0u8; rows * qdot::col_bytes(GgmlType::Q3_K, embd)],
            gate_up: std::array::from_fn(|_| Tensor2::zeros(ff, rows)),
            pars: std::array::from_fn(|_| Tensor2::zeros(ff, rows)),
            downs: (0..distinct).map(|_| Tensor2::zeros(embd, rows)).collect(),
            out: vec![0.0; embd * rows],
            plan: R8Plan::with_room(rows, n_used),
            lists: vec![vec![(0, 0.0); n_used]; rows],
        }
    }
}

/// The `union` shape's blocks, made before the round's first token: the rows'
/// activation blocks ([`union_xs`]), one chunk's gate, up, combine and down
/// blocks of `rows` columns, the store of every chunk's downs but the last,
/// `x` quantized, the plan, the output and the rows' lists.
struct ChunkBlocks {
    xs: Vec<Tensor2>,
    gate_up: [Tensor2; 2 * UNION_CHUNK],
    pars: [Tensor2; UNION_CHUNK],
    downs: [Tensor2; UNION_CHUNK],
    store: Vec<f32>,
    xq: QuantizedCols,
    plan: R8Plan,
    out: Vec<f32>,
    lists: Vec<Vec<(u32, f32)>>,
}

impl ChunkBlocks {
    /// Blocks for calls over `rows` columns of up to `n_used` experts.
    fn new(xs: &[Tensor2], rows: usize, embd: usize, ff: usize, n_used: usize) -> ChunkBlocks {
        ChunkBlocks {
            xs: union_xs(xs, rows, embd),
            gate_up: std::array::from_fn(|_| Tensor2::zeros(ff, rows)),
            pars: std::array::from_fn(|_| Tensor2::zeros(ff, rows)),
            downs: std::array::from_fn(|_| Tensor2::zeros(embd, rows)),
            store: vec![0.0; embd * n_used * rows],
            xq: QuantizedCols::new(),
            plan: R8Plan::with_room(rows, n_used),
            out: vec![0.0; embd * rows],
            lists: vec![vec![(0, 0.0); n_used]; rows],
        }
    }
}

/// The chunked union flow over activation block `xi` and the blocks' lists
/// (their first `n_host` entries): the distinct experts in ascending id in
/// chunks of [`UNION_CHUNK`], per chunk one `ops::matmul_q_group_cols_into`
/// for its gates and ups over the columns that list each (reading `x`
/// quantized once, over the pool, when the call is wider than
/// `ops::DEFER_MAX_COLS` columns) and one for its downs over the combines
/// (the layer's clamp), each chunk's downs but the last's copied to the store
/// ([`stash_downs`]), then each column's sum from zero in its list order. The
/// flow the host tier ran before its five-dispatch union, rebuilt from the
/// group entry: its output is `HostLayer::experts_union_into`'s, bit for
/// bit. Allocates nothing.
fn union_chunks_call(
    host: &HostLayer,
    split: &Split,
    l: &Layer<'_>,
    xi: usize,
    n_host: usize,
    u: &mut ChunkBlocks,
) -> Result<(), BenchError> {
    let ChunkBlocks {
        xs,
        gate_up,
        pars,
        downs,
        store,
        xq,
        plan,
        out,
        lists,
    } = u;
    let x = xs[xi].view();
    let (embd, k) = (x.ne0(), x.ne1());
    plan.build(lists, n_host, l)?;
    let plan = &*plan;
    let stacks = host.stacks();
    let limit = host.swiglu_limit();
    let resolve = |d: usize| -> Result<[Weight<'_>; 3], ModelError> {
        let e = plan.ids[d] as usize;
        Ok([
            stacks[GATE].expert(split, e)?,
            stacks[UP].expert(split, e)?,
            stacks[DOWN].expert(split, e)?,
        ])
    };
    let mut filled = false;
    if k > DEFER_MAX_COLS && plan.n() > 0 {
        filled = xq.fill(resolve(0)?[GATE].ty(), x, k);
    }
    let xq: Option<&QuantizedCols> = filled.then_some(&*xq);
    let n_chunks = plan.n().div_ceil(UNION_CHUNK);
    let per = if n_chunks == 0 {
        0
    } else {
        plan.n().div_ceil(n_chunks)
    };
    for c in 0..n_chunks {
        let (a, b) = (c * per, ((c + 1) * per).min(plan.n()));
        let n = b - a;
        // The unused tails keep the chunk's first expert's gate; only the
        // first n (2n) are passed.
        let first = resolve(a)?[GATE];
        let mut gu = [first; 2 * UNION_CHUNK];
        let mut down = [first; UNION_CHUNK];
        let mut srcs: [GroupInput<'_>; 2 * UNION_CHUNK] =
            [GroupInput::Cols(x, &[]); 2 * UNION_CHUNK];
        for i in 0..n {
            let [g, up, dw] = resolve(a + i)?;
            (gu[2 * i], gu[2 * i + 1], down[i]) = (g, up, dw);
            let cols = plan.cols(a + i);
            let src = |w: &Weight<'_>| match xq {
                Some(q) if q.serves(w.ty(), w.k()) => GroupInput::Quantized(x, q, cols),
                _ => GroupInput::Cols(x, cols),
            };
            srcs[2 * i] = src(&g);
            srcs[2 * i + 1] = src(&up);
            gate_up[2 * i].set_cols(cols.len());
            gate_up[2 * i + 1].set_cols(cols.len());
            pars[i].set_cols(cols.len());
            downs[i].set_cols(cols.len());
        }
        matmul_q_group_cols_into(
            "host_gate_up",
            &gu[..2 * n],
            &srcs[..2 * n],
            &mut gate_up[..2 * n],
            &mut [],
        )?;
        let combines: [GroupInput<'_>; UNION_CHUNK] = std::array::from_fn(|i| {
            if i < n {
                GroupInput::SwigluClamp(&gate_up[2 * i], &gate_up[2 * i + 1], limit)
            } else {
                GroupInput::Cols(x, &[])
            }
        });
        matmul_q_group_cols_into(
            "host_down",
            &down[..n],
            &combines[..n],
            &mut downs[..n],
            &mut pars[..n],
        )?;
        if c + 1 < n_chunks {
            stash_downs(&downs[..n], &plan.off[a..=b], embd, store);
        }
    }
    let (downs, store, lists) = (&*downs, &*store, &*lists);
    let last = n_chunks.saturating_sub(1) * per;
    // Column `j`'s weighted sum into `o`, from zero in the order of its list:
    // the last chunk's downs from their blocks, the others' from the store.
    let sum_col = |j: usize, o: &mut [f32]| {
        o.fill(0.0);
        for &(e, w) in &lists[j][..n_host] {
            let d = plan.index_of(e);
            let t = plan
                .cols(d)
                .binary_search(&j)
                .expect("a listing column is in its expert's columns");
            let col = if d >= last {
                downs[d - last].col(t)
            } else {
                let q = plan.off[d] + t;
                &store[q * embd..(q + 1) * embd]
            };
            for (o, &dv) in o.iter_mut().zip(col) {
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

/// A chunk's down columns into the store before the next chunk reuses their
/// blocks: block `i`'s columns land at store columns `off[i]..off[i + 1]`
/// (`off` is the plan's offsets of the chunk's experts, one past its last).
/// Columns up to [`UNION_INLINE_COLS`] copy on the caller, more split across
/// the pool.
fn stash_downs(blocks: &[Tensor2], off: &[usize], embd: usize, store: &mut [f32]) {
    let (q0, q1) = (off[0], off[off.len() - 1]);
    if q1 - q0 <= UNION_INLINE_COLS {
        for (blk, w) in blocks.iter().zip(off.windows(2)) {
            store[w[0] * embd..w[1] * embd].copy_from_slice(&blk.data[..(w[1] - w[0]) * embd]);
        }
        return;
    }
    let dst = Cells(store.as_mut_ptr());
    // SAFETY (construction site): each participant writes only store columns
    // in its own chunk of `q0..q1`; the chunks partition them, and the join
    // publishes the writes.
    threads::pool().for_each_chunk(q1 - q0, |qs| {
        for q in q0 + qs.start..q0 + qs.end {
            // The block whose range holds `q`: `off` ascends.
            let i = off.partition_point(|&o| o <= q) - 1;
            let col = blocks[i].col(q - off[i]);
            // SAFETY: store column `q` is in this participant's chunk — see the construction site.
            let cell = unsafe { std::slice::from_raw_parts_mut(dst.ptr().add(q * embd), embd) };
            cell.copy_from_slice(col);
        }
    });
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
#[allow(clippy::too_many_arguments)] // one call's whole input: the layer three ways, its tile, its lists' parts, the blocks and the tally
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
    fill_lists(&mut u.lists, l, slots, n_host)?;
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
        claim_block,
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
            gate_up[2 * i].set_cols(m);
            gate_up[2 * i + 1].set_cols(m);
            pars[i].set_cols(m);
            downs[a + i].set_cols(m);
        }
        let claim = (narrow_call && ch == 0).then_some(&claims);
        let empty: &[u8] = &[];
        let mut mats = [[empty; 2]; UNION_CHUNK];
        for (i, m) in mats.iter_mut().enumerate().take(n) {
            let s = plan.slots[a + i];
            *m = match tile {
                GateUpTile::RowLane(r8) => [r8[s][0].as_slice(), r8[s][1].as_slice()],
                GateUpTile::Column => {
                    let [g, up] = [GATE, UP].map(|m| l.weight(s, m));
                    for (w, m) in [(g, GATE), (up, UP)] {
                        if w.ty() != GgmlType::Q3_K || w.k() != embd {
                            return Err(format!(
                                "unionq: layer {} expert {} {} is {:?} over {} values; the column \
                                 tile takes Q3_K over {embd}",
                                l.index,
                                plan.ids[a + i],
                                MATRIX[m],
                                w.ty(),
                                w.k()
                            )
                            .into());
                        }
                    }
                    [g.bytes(), up.bytes()]
                }
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
            (claim, *claim_block),
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
/// order off one counter `block` units at a time; `mats[i]` holds the
/// chunk's expert `i`'s gate and up, in the row-lane layout (`row_lane`: each
/// run of up to `qdot::TILE_COLS` of the expert's columns goes to
/// `qdot::dot_q3k_r8_cols` once for the eight rows) or as the file's Q3_K
/// rows over `x`'s width (each of the eight rows through `qdot::dot_row_cols`
/// per run, as the union call's row dispatch does). The values land in
/// `outs[2i]` (gate) and `outs[2i + 1]` (up), already narrowed to the
/// expert's columns. With `claims`, the participants first quantize `x` into
/// `xq`; without, `xq` already holds it. A chunk of no expert, rows that are
/// not whole groups of eight and a matrix of other bytes than its rows are
/// named errors; a kernel error stops the unit walk and is returned.
#[allow(clippy::too_many_arguments)] // one chunk's whole dispatch: plan and offset, the weights and their layout, x and its bytes, the claims and their block, the outputs
fn r8_gate_up(
    plan: &R8Plan,
    mats: &[[&[u8]; 2]],
    row_lane: bool,
    x: &Tensor2,
    xq: &mut [u8],
    cb: usize,
    (claims, block): (Option<&QuantClaims>, usize),
    a: usize,
    outs: &mut [Tensor2],
) -> Result<(), BenchError> {
    const R8: usize = qdot::Q3K_R8_ROWS;
    let embd = x.ne0;
    let ff = outs
        .first()
        .map(|t| t.ne0)
        .ok_or("r8_gate_up: a chunk of no expert")?;
    if !ff.is_multiple_of(R8) {
        return Err(format!("r8_gate_up: {ff} rows are not whole groups of {R8}").into());
    }
    let rb = mats[0][0].len() / ff;
    if let Some((i, m)) = (0..mats.len())
        .flat_map(|i| [(i, 0), (i, 1)])
        .find(|&(i, m)| mats[i][m].len() != ff * rb)
    {
        return Err(format!(
            "r8_gate_up: expert {i}'s {} is {} bytes, not {ff} rows of {rb}",
            MATRIX[m],
            mats[i][m].len()
        )
        .into());
    }
    // The raw writes below rest on these two: a gate and an up block per
    // expert, each narrowed to its expert's columns.
    assert_eq!(
        outs.len(),
        2 * mats.len(),
        "r8_gate_up: a gate and an up block per expert"
    );
    for (p, t) in outs.iter().enumerate() {
        let m = plan.cols(a + p / 2).len();
        assert_eq!(
            (t.ne0, t.ne1, t.data.len()),
            (ff, m, ff * m),
            "r8_gate_up: block {p} holds its expert's columns"
        );
    }
    assert!(block > 0, "r8_gate_up: a claim takes at least one unit");
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
            let u0 = next.fetch_add(block, Ordering::Relaxed);
            if u0 >= units {
                return;
            }
            for u in u0..units.min(u0 + block) {
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
                                    std::ptr::copy_nonoverlapping(
                                        vals.as_ptr(),
                                        out.ptr().add(at),
                                        R8,
                                    );
                                }
                            }
                        } else {
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
    lanes: Lanes,
    l: &Layer<'_>,
    slots: &[usize],
    x: &Tensor2,
    b: &mut Blocks,
    tally: &mut Tally,
) -> Result<(), ModelError> {
    engine_rows_layer(lanes, l, slots, &[x], b, tally)
}

/// Probe shape for a k-token step: `xs.len()` rows, row `i` owning the
/// `slots.len() / xs.len()` slots `slots[i * n..(i + 1) * n]` and reading
/// `xs[i]`. Every row's gate and up in one group, every row's down in one
/// group — the dispatches a tier that takes k tokens per call would issue.
fn engine_rows_layer(
    lanes: Lanes,
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
    matmul_q_group_into_on(lanes, "host_gate_up", &ws, &xin, &mut b.gu[..2 * n])?;
    tally.add("gate+up", bytes_of(&ws), t0);
    let dw: Vec<Weight<'_>> = slots.iter().map(|&s| l.weight(s, DOWN)).collect();
    let (pairs, _) = b.gu[..2 * n].as_chunks::<2>();
    let srcs: Vec<GroupInput<'_>> = pairs
        .iter()
        .map(|[g, u]| GroupInput::Swiglu(g, u))
        .collect();
    let t0 = Instant::now();
    matmul_q_group_swiglu_into_on(
        lanes,
        "host_down",
        &dw,
        &srcs,
        &mut b.down[..n],
        &mut b.par[..n],
    )?;
    tally.add("down", bytes_of(&dw), t0);
    Ok(())
}

/// A warm job: every participant's [`warm_share`] of `spans` over `map`, one
/// pool dispatch of an index a participant, run to the end (no stop). Tallied
/// as the `warm` kind under the bytes its walks reached, so that dispatch
/// line's rate is the warm's; returns those bytes.
fn warm_job(spans: &[WarmSpan<'_>], map: CcdMap, tally: &mut Tally) -> u64 {
    let reached = AtomicU64::new(0);
    let t0 = Instant::now();
    threads::pool().for_each_chunk(map.threads(), |r| {
        let done = warm_share(spans, &map, r.start, || false);
        reached.fetch_add(done.bytes, Ordering::Relaxed);
    });
    let bytes = reached.into_inner();
    tally.add("warm", bytes, t0);
    bytes
}

/// The matrices of the first `k` of `slots`' experts as the engine shape's
/// dispatches read them: each expert's gate, up and down, with their rows as
/// the units.
fn engine_spans<'a>(l: &Layer<'a>, slots: &[usize], k: usize) -> Vec<WarmSpan<'a>> {
    slots
        .iter()
        .take(k)
        .flat_map(|&s| {
            [GATE, UP, DOWN].map(|m| {
                let w = l.weight(s, m);
                WarmSpan {
                    bytes: w.bytes(),
                    units: w.n(),
                }
            })
        })
        .collect()
}

/// The `engine-warm` shape: the first `k` experts of the token warmed, then
/// the engine shape's two dispatches on CCD-major lanes.
fn engine_warm_layer(
    map: CcdMap,
    l: &Layer<'_>,
    slots: &[usize],
    k: usize,
    x: &Tensor2,
    b: &mut Blocks,
    tally: &mut Tally,
) -> Result<(), ModelError> {
    if k > 0 {
        warm_job(&engine_spans(l, slots, k), map, tally);
    }
    engine_layer(Lanes::Ccd(map), l, slots, x, b, tally)
}

/// The first `k` distinct expert ids of `lists`' first `n_host` entries.
fn first_distinct_ids(lists: &[Vec<(u32, f32)>], n_host: usize, k: usize) -> Vec<u32> {
    let mut ids: Vec<u32> = Vec::with_capacity(k);
    for &(id, _) in lists.iter().flat_map(|r| &r[..n_host]) {
        if ids.len() < k && !ids.contains(&id) {
            ids.push(id);
        }
    }
    ids
}

/// The `union5-warm` shape: the first `k` distinct experts of the token's lists
/// warmed ([`HostLayer::warm_spans`]), then the union call on the host tier's
/// CCD-major lanes ([`HostLayer::experts_step_union_into`]), tallied whole as
/// `union5` is.
#[allow(clippy::too_many_arguments)] // one call's whole input: the layer three ways, its lists' parts, the warm, the blocks and the tally
fn union5_warm_layer(
    map: CcdMap,
    host: &HostLayer,
    split: &Split,
    l: &Layer<'_>,
    slots: &[usize],
    (n_host, distinct, k): (usize, usize, usize),
    u: &mut UnionBlocks,
    x: usize,
    tally: &mut Tally,
) -> Result<(), BenchError> {
    fill_lists(&mut u.lists, l, slots, n_host)?;
    if k > 0 {
        let ids = first_distinct_ids(&u.lists, n_host, k);
        let mut spans = Vec::new();
        host.warm_spans(R8Source::rows(split), &ids, &mut spans)?;
        warm_job(&spans, map, tally);
    }
    let lists: Vec<&[(u32, f32)]> = u.lists.iter().map(|r| &r[..n_host]).collect();
    let t0 = Instant::now();
    host.experts_step_union_into(
        R8Source::rows(split),
        &u.xs[x],
        &lists,
        &mut u.out,
        &mut u.scratch,
    )?;
    tally.add("union5", distinct as u64 * l.expert_bytes(), t0);
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
fn print_table(path: &str, split: &Split, meta: &Meta, family: &Family, layers: &[Layer<'_>]) {
    println!(
        "v41host model={path} family={} shards={} blocks={} embd={} ff={} experts={} used={} working_set={WORKING_SET}",
        family.name(),
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

/// The activation a fused kernel for one weight type reads, decoded exactly.
struct KernelAct {
    values: Vec<f64>,
    /// For q5_1, each 32-value block's stored `d · Σq`, the factor the
    /// kernel's min term multiplies the weight block's min by; empty for
    /// every other type.
    block_sums: Vec<f64>,
}

/// The activation a fused kernel for weight type `ty` reads: `x` through
/// `qdot::quantize_col`, the bytes decoded exactly in f64. A type with no
/// decoder here is the named error.
fn kernel_activation(ty: GgmlType, x: &[f32]) -> Result<KernelAct, BenchError> {
    let mut col = vec![0u8; qdot::col_bytes(ty, x.len())];
    qdot::quantize_col(ty, x, &mut col);
    let (values, block_sums) = match ty {
        GgmlType::Q3_K => (decode_q8k(&col), Vec::new()),
        GgmlType::Q4_K | GgmlType::Q5_K => (decode_q82x4(&col), Vec::new()),
        GgmlType::Q5_1 => decode_q5_1(&col, x.len())?,
        GgmlType::Q8_0 => (decode_q8_0(&col, x.len())?, Vec::new()),
        other => return Err(format!("no activation decoder for {other}").into()),
    };
    if values.len() != x.len() {
        return Err(format!("{ty}: decoded {} values of {}", values.len(), x.len()).into());
    }
    Ok(KernelAct { values, block_sums })
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

/// One 32-value block of the q8_2 family as `qdot::quantize_col` writes it:
/// its bf16 scale, its stored code sum and its codes.
struct Q82Block {
    d: f64,
    sum: i16,
    codes: [i8; 32],
}

/// The blocks of a q8_2-family column of `k` values (q5_1 and q8_0 weights):
/// `k / 128` block_q8_2_x4 groups of 144 bytes — four bf16 scales at 0, four
/// i16 code sums at 8, four runs of 32 codes from 16 — then one 36-byte
/// block_q8_2 per 32 values left: the bf16 scale at 0, the i16 code sum at 2,
/// the codes from 4.
fn q82_blocks(col: &[u8], k: usize) -> Result<Vec<Q82Block>, BenchError> {
    let (groups, tails) = (k / 128, (k % 128) / 32);
    if !k.is_multiple_of(32) || col.len() != groups * 144 + tails * 36 {
        return Err(format!(
            "a q8_2 column of {} bytes is not {k} values in x4 groups and tails",
            col.len()
        )
        .into());
    }
    let bf16 = |b: &[u8]| {
        f64::from(f32::from_bits(
            u32::from(u16::from_le_bytes([b[0], b[1]])) << 16,
        ))
    };
    let sum = |b: &[u8]| i16::from_le_bytes([b[0], b[1]]);
    let codes = |b: &[u8]| -> [i8; 32] { std::array::from_fn(|i| i8::from_le_bytes([b[i]])) };
    let (x4, tail) = col.split_at(groups * 144);
    let mut out = Vec::with_capacity(k / 32);
    for g in x4.as_chunks::<144>().0 {
        for ir in 0..4 {
            out.push(Q82Block {
                d: bf16(&g[2 * ir..]),
                sum: sum(&g[8 + 2 * ir..]),
                codes: codes(&g[16 + 32 * ir..]),
            });
        }
    }
    for t in tail.as_chunks::<36>().0 {
        out.push(Q82Block {
            d: bf16(&t[..]),
            sum: sum(&t[2..]),
            codes: codes(&t[4..]),
        });
    }
    Ok(out)
}

/// A q8_0 kernel's activation: every value `d · q` of its q8_2 blocks.
fn decode_q8_0(col: &[u8], k: usize) -> Result<Vec<f64>, BenchError> {
    Ok(q82_blocks(col, k)?
        .iter()
        .flat_map(|b| b.codes.iter().map(move |&q| b.d * f64::from(q)))
        .collect())
}

/// A q5_1 kernel's activation: every value `d · q` of its q8_2 blocks, and
/// each block's stored sum `d · Σq`, the min term's factor.
fn decode_q5_1(col: &[u8], k: usize) -> Result<(Vec<f64>, Vec<f64>), BenchError> {
    let blocks = q82_blocks(col, k)?;
    let values = blocks
        .iter()
        .flat_map(|b| b.codes.iter().map(move |&q| b.d * f64::from(q)))
        .collect();
    let sums = blocks.iter().map(|b| b.d * f64::from(b.sum)).collect();
    Ok((values, sums))
}

/// The part of a q5_1 row's dot its kernel takes from the activation's stored
/// block sums: `Σ_b m_b · (s_b − Σ_{i∈b} a_i)`, `m_b` the weight block's min
/// (the f16 at bytes 2..4 of its 24), `s_b` the stored sum. Zero when every
/// stored sum is its block's values' sum; a missing sum is the named error.
fn q5_1_min_term(row: &[u8], act: &KernelAct) -> Result<f64, BenchError> {
    let blocks = act.values.len() / 32;
    if act.block_sums.len() != blocks || row.len() != blocks * 24 {
        return Err(format!(
            "q5_1: {} activation block sums and {} row bytes for {} values",
            act.block_sums.len(),
            row.len(),
            act.values.len()
        )
        .into());
    }
    Ok(row
        .as_chunks::<24>()
        .0
        .iter()
        .zip(act.values.as_chunks::<32>().0.iter().zip(&act.block_sums))
        .map(|(w, (a, &s))| {
            let m = f64::from(half_to_f32(u16::from_le_bytes([w[2], w[3]])));
            m * (s - a.iter().sum::<f64>())
        })
        .sum())
}

/// A column of `k` values for the decoders' round trip: [`activations`]'
/// draw, the second 32-value block all zeros, the others scaled by 1 to 16
/// so neighbouring blocks' scales differ.
fn decoder_column(k: usize) -> Vec<f32> {
    let mut rng = Rng::new(&[KEY_X, k as u64]);
    (0..k)
        .map(|i| {
            let u = (rng.next_u64() >> 40) as f32 / 8_388_608.0 - 1.0;
            let block = i / 32;
            let scale = (1u32 << (block % 5)) as f32;
            let outlier = if i.is_multiple_of(61) { 8.0 } else { 1.0 };
            if block == 1 { 0.0 } else { u * scale * outlier }
        })
        .collect()
}

/// Width of the decoders' round trip besides a stack's own `k`: one x4
/// group and three block_q8_2 tails.
const DECODER_TAIL_K: usize = 224;

/// The q8_2-family decoders ([`decode_q5_1`], [`decode_q8_0`]) against the
/// input `qdot::quantize_col` encoded, for every such type a checked layer's
/// stack has, at the stack's `k` and at [`DECODER_TAIL_K`]: every decoded
/// value within half a step of its input (the block's scale over two, plus
/// the f32 rounding of `x · (1/d)` at a code of at most 127.5, under
/// `1e-4 · d`); q5_1's stored block sums equal to its values' sums exactly
/// (both are `d` times an integer under 2^13, exact in f64), and q8_0 with
/// none.
fn check_decoders(
    c: &mut Checks,
    layers: &[Layer<'_>],
    covered: &[usize],
) -> Result<(), BenchError> {
    let mut runs: Vec<(GgmlType, usize)> = Vec::new();
    for &li in covered {
        for m in [GATE, UP, DOWN] {
            let v = layers[li].view(0, m);
            if matches!(v.ty, GgmlType::Q5_1 | GgmlType::Q8_0) {
                for k in [usize::try_from(v.dims[0])?, DECODER_TAIL_K] {
                    if !runs.contains(&(v.ty, k)) {
                        runs.push((v.ty, k));
                    }
                }
            }
        }
    }
    for (ty, k) in runs {
        let x = decoder_column(k);
        let act = kernel_activation(ty, &x)?;
        let mut col = vec![0u8; qdot::col_bytes(ty, k)];
        qdot::quantize_col(ty, &x, &mut col);
        let blocks = q82_blocks(&col, k)?;
        let (mut within, mut worst) = (true, 0.0f64);
        for (i, (&v, &xi)) in act.values.iter().zip(&x).enumerate() {
            let d = blocks[i / 32].d;
            let err = (v - f64::from(xi)).abs();
            within &= err <= 0.5 * d + 1e-4 * d;
            if d > 0.0 {
                worst = worst.max(err / d);
            }
        }
        let sums_exact = if ty == GgmlType::Q5_1 {
            act.block_sums.len() == k / 32
                && act
                    .values
                    .as_chunks::<32>()
                    .0
                    .iter()
                    .zip(&act.block_sums)
                    .all(|(vals, &s)| s == vals.iter().sum::<f64>())
        } else {
            act.block_sums.is_empty()
        };
        let line = format!(
            "check decoder={ty} k={k} blocks={} values={} max_err_steps={worst:.4} step_band=0.5001 \
             block_sums_exact={sums_exact}",
            blocks.len(),
            act.values.len()
        );
        c.record(
            format!("decoder {ty} k {k}"),
            &line,
            within && sums_exact && act.values.len() == k,
        );
    }
    Ok(())
}

/// f64 dot products of rows `rows` of `w` (read from its shard `g`) with `act`.
fn reference(
    g: &Gguf,
    w: &TensorInfo,
    act: &KernelAct,
    rows: &[usize],
) -> Result<Vec<f64>, BenchError> {
    let k = usize::try_from(w.dims[0])?;
    let n = usize::try_from(w.dims[1])?;
    if act.values.len() != k {
        return Err(format!(
            "{}: {} activation values for k = {k}",
            w.name,
            act.values.len()
        )
        .into());
    }
    let bytes = g.data(w)?;
    let rb = bytes.len() / n;
    let mut vals = vec![0.0f32; k];
    rows.iter()
        .map(|&r| -> Result<f64, BenchError> {
            let row = &bytes[r * rb..(r + 1) * rb];
            dequant_row(w.ty, row, &mut vals)?;
            let dot: f64 = vals
                .iter()
                .zip(&act.values)
                .map(|(&v, &a)| f64::from(v) * a)
                .sum();
            let min = if w.ty == GgmlType::Q5_1 {
                q5_1_min_term(row, act)?
            } else {
                0.0
            };
            Ok(dot + min)
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
    act: &KernelAct,
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

/// The union shapes against the engine shape on layer `l`: two rows over
/// `xs[li % N_X]` and the next column, row 0 listing `slots`, row 1 the same
/// set shifted by one with one working-set expert neither lists. Each row's
/// output of the `union` and the `union5` call must equal the list-order sum
/// from zero of the engine shape's downs for that row, at weight `1 / n`, bit
/// for bit.
fn check_union(
    c: &mut Checks,
    l: &Layer<'_>,
    host: &HostLayer,
    split: &Split,
    slots: &[usize],
    xs: &[Tensor2],
    n_used: usize,
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
    let mut u = ChunkBlocks::new(xs, 2, embd, ff, n_used);
    let mut u5 = UnionBlocks::new(xs, 2, embd, ff, n_used)?;
    let x0 = l.index % N_X;
    let mut tally = Tally::default();
    union_layer(host, split, l, &rows, n, n + 1, &mut u, x0, &mut tally)?;
    union5_layer(host, split, l, &rows, n, n + 1, &mut u5, x0, &mut tally)?;
    let w = 1.0 / n as f32;
    let (mut same, mut same5) = (true, true);
    for (i, run) in rows.chunks_exact(n).enumerate() {
        let mut b = Blocks::new(std::slice::from_ref(l), n)?;
        engine_layer(Lanes::Flat, l, run, &xs[(x0 + i) % N_X], &mut b, &mut tally)?;
        let mut want = vec![0.0f32; embd];
        for d in &b.down[..n] {
            for (o, &dv) in want.iter_mut().zip(d.col(0)) {
                *o += w * dv;
            }
        }
        same &= bits_equal(&u.out[i * embd..(i + 1) * embd], &want);
        same5 &= bits_equal(&u5.out[i * embd..(i + 1) * embd], &want);
    }
    for (shape, same) in [("union", same), ("union5", same5)] {
        let line = format!(
            "check layer={} shape={shape} rows=2 experts={n} distinct={} outputs_bits_equal_engine_sum={same}",
            l.index,
            n + 1
        );
        c.record(format!("layer {} {shape}", l.index), &line, same);
    }
    Ok(())
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
    let n = slots.len();
    let mut b = Blocks::new(std::slice::from_ref(l), n)?;
    engine_layer(Lanes::Flat, l, slots, x, &mut b, &mut tally)?;
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

/// The `union5`, `unionr8` and `unionq` shapes against the union shape on
/// layer `li`, bit for bit, for one token of `rows` rows walking a pool of
/// `distinct` working-set slots, `n_host` each, as an arm's rows do.
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
    let mut u = ChunkBlocks::new(&bench.xs, rows, embd, ff, bench.n_used);
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
    let mut u5 = UnionBlocks::new(&bench.xs, rows, embd, ff, bench.n_used)?;
    union5_layer(
        host,
        bench.split,
        l,
        &slots,
        n_host,
        distinct_n,
        &mut u5,
        x0,
        &mut tally,
    )?;
    let mut outs = vec![("union5", u5.out)];
    for (shape, tile) in [
        ("unionr8", GateUpTile::RowLane(r8)),
        ("unionq", GateUpTile::Column),
    ] {
        let mut v = R8Blocks::new(&bench.xs, (rows, embd, ff), bench.n_used, distinct_n, 1);
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
        outs.push((shape, v.out));
    }
    for (shape, out) in outs {
        let same = bits_equal(&u.out, &out);
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

/// The fewest and the most columns one distinct expert serves in the draw
/// `ids` over `layers`: the arm's columns per expert as its tokens read them,
/// counted from the draw rather than from the arm's arithmetic.
fn drawn_cols(ids: &[Vec<usize>], layers: &[usize]) -> (usize, usize) {
    let mut counts = [0usize; WORKING_SET];
    let (mut lo, mut hi) = (usize::MAX, 0);
    for &l in layers {
        counts.fill(0);
        for &s in &ids[l] {
            counts[s] += 1;
        }
        for &n in counts.iter().filter(|&&n| n > 0) {
            (lo, hi) = (lo.min(n), hi.max(n));
        }
    }
    (lo.min(hi), hi)
}

/// Before any timing (every layer), and under `--check` (the checked layers,
/// or every layer for `--arms`): one token of every `union5`, `unionr8` and
/// `unionq` arm through its shape and the union shape, layer by layer over
/// `layers`, whose outputs must be equal bit for bit; a mismatch is the named
/// error.
fn check_r8_arms(bench: &Bench<'_>, arms: &[Arm], layers: &[usize]) -> Result<(), BenchError> {
    let embd = bench.xs.first().ok_or("no activation columns")?.ne0;
    for arm in arms
        .iter()
        .filter(|a| matches!(a.shape, Shape::Union5 | Shape::UnionR8 | Shape::UnionQ))
    {
        let ids = bench.draw(*arm, 0, 0);
        let mut differ = Vec::new();
        for &l in layers {
            let layer = &bench.layers[l];
            let host = bench
                .hosts
                .get(l)
                .ok_or("no host layer for the union arm")?;
            let ff = layer.weight(0, GATE).n();
            let x = l % N_X;
            let mut tally = Tally::default();
            let mut u = ChunkBlocks::new(&bench.xs, arm.rows, embd, ff, bench.n_used);
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
            let out = if arm.shape == Shape::Union5 {
                let mut v = UnionBlocks::new(&bench.xs, arm.rows, embd, ff, bench.n_used)?;
                union5_layer(
                    host,
                    bench.split,
                    layer,
                    &ids[l],
                    arm.n_host,
                    arm.union,
                    &mut v,
                    x,
                    &mut tally,
                )?;
                v.out
            } else {
                let tile = bench.gate_up_tile(arm.shape, l)?;
                let mut v = R8Blocks::new(
                    &bench.xs,
                    (arm.rows, embd, ff),
                    bench.n_used,
                    arm.union,
                    arm.claim_block,
                );
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
                v.out
            };
            if !bits_equal(&u.out, &out) {
                differ.push(layer.index);
            }
        }
        let (lo, hi) = drawn_cols(&ids, layers);
        println!(
            "check arm={} layers={} distinct={} cols_per_expert={:.2} cols_drawn={lo}..{hi} \
             outputs_bits_equal_union={}",
            arm.label(),
            layers.len(),
            arm.union,
            arm.cols_per_expert(),
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

/// Before any timing (every layer), and under `--check` (the checked
/// layers, or every layer for `--arms`): one token of every `unioncard` arm
/// through its shape, layer by layer over `layers`, against the one-column
/// card-rule calls — each row's output the list-order sum from zero of
/// `w · down(h(gate·up·SwiGLU))` its list names, every expert computed one
/// column at a time, bit for bit. A mismatch is the named error.
fn check_card_arms(bench: &Bench<'_>, arms: &[Arm], layers: &[usize]) -> Result<(), BenchError> {
    let embd = bench.xs.first().ok_or("no activation columns")?.ne0;
    for arm in arms.iter().filter(|a| matches!(a.shape, Shape::UnionCard)) {
        let ids = bench.draw(*arm, 0, 0);
        let mut differ = Vec::new();
        let mut skipped = 0usize;
        for &l in layers {
            let layer = &bench.layers[l];
            if layer.stacks[DOWN].info.ty != GgmlType::Q4_K {
                skipped += 1;
                continue;
            }
            let ff = layer.weight(0, GATE).n();
            let x = l % N_X;
            let mut tally = Tally::default();
            let mut c = CardBlocks::new(&bench.xs, arm.rows, embd, ff, arm.n_host);
            unioncard_layer(layer, &ids[l], arm.n_host, &mut c, x, &mut tally)?;
            // The reference: one column at a time, on the pool.
            let n = arm.rows * arm.n_host;
            let mut downs = vec![0.0f32; n * embd];
            {
                let slots = &ids[l];
                let downs = std::sync::Mutex::new(&mut downs);
                threads::pool().for_each_chunk(n, |range| {
                    for s in range {
                        let row = s / arm.n_host;
                        let wslot = slots[s];
                        let xcol = qdot::card_q8_1(&bench.xs[(x + row) % N_X].data);
                        let (rb_gu, rb_d) = (
                            layer.weight(wslot, GATE).bytes().len() / ff,
                            layer.weight(wslot, DOWN).bytes().len() / embd,
                        );
                        let (gb, ub, db) = (
                            layer.weight(wslot, GATE).bytes(),
                            layer.weight(wslot, UP).bytes(),
                            layer.weight(wslot, DOWN).bytes(),
                        );

                        let mut hcol = vec![0.0f32; ff];
                        for r in 0..ff {
                            let g =
                                qdot::card_q3k_dot_row(&gb[r * rb_gu..][..rb_gu], &xcol).unwrap();
                            let u =
                                qdot::card_q3k_dot_row(&ub[r * rb_gu..][..rb_gu], &xcol).unwrap();
                            hcol[r] = qdot::card_swiglu_clamp_1(g, u, 0.0);
                        }
                        let hq = qdot::card_q8_1(&hcol);
                        let mut out = downs.lock().unwrap();
                        for r in 0..embd {
                            out[s * embd + r] =
                                qdot::card_q4k_dot_row(&db[r * rb_d..][..rb_d], &hq).unwrap();
                        }
                    }
                });
            }
            let w = 1.0 / arm.n_host as f32;
            let mut ok = true;
            for row in 0..arm.rows {
                let mut want = vec![0.0f32; embd];
                for j in 0..arm.n_host {
                    let dv = &downs[(row * arm.n_host + j) * embd..][..embd];
                    for (o, &v) in want.iter_mut().zip(dv) {
                        *o += w * v;
                    }
                }
                ok &= bits_equal(&c.out[row * embd..(row + 1) * embd], &want);
            }
            if !ok {
                differ.push(layer.index);
            }
        }
        let (lo, hi) = drawn_cols(&ids, layers);
        println!(
            "check arm={} layers={} skipped_non_q4k_down={skipped} distinct={} \
             cols_per_expert={:.2} cols_drawn={lo}..{hi} outputs_bits_equal_one_column={}",
            arm.label(),
            layers.len() - skipped,
            arm.union,
            arm.cols_per_expert(),
            differ.is_empty()
        );
        if !differ.is_empty() {
            return Err(format!(
                "arm {}: its output differs from the one-column card-rule calls on layers {differ:?}",
                arm.label()
            )
            .into());
        }
    }
    Ok(())
}

/// Before any timing (every layer), and under `--check` (the checked layers, or
/// every layer for `--arms`): one token of every warm arm over `layers`, held
/// to two things. Its warm job reaches exactly `w<k>` experts' bytes — the sum
/// of every participant's [`warm_share`] is `k` times the layer's expert bytes,
/// so a CCD that warms nothing, or a line walked twice or never, is red — and
/// its CCD-major leg writes the flat leg's output bit for bit. A miss is the
/// named error.
fn check_warm_arms(bench: &Bench<'_>, arms: &[Arm], layers: &[usize]) -> Result<(), BenchError> {
    let embd = bench.xs.first().ok_or("no activation columns")?.ne0;
    for arm in arms.iter().filter(|a| a.shape.warms()) {
        let map = bench.ccd.ok_or(NO_CCD_MAP)?;
        let ids = bench.draw(*arm, 0, 0);
        let (mut short, mut differ) = (Vec::new(), Vec::new());
        for &l in layers {
            let layer = &bench.layers[l];
            let slots = &ids[l];
            let want = arm.warm as u64 * layer.expert_bytes();
            let x = l % N_X;
            let mut tally = Tally::default();
            let (reached, same) = if arm.shape == Shape::EngineWarm {
                let spans = engine_spans(layer, slots, arm.warm);
                let reached = warm_job(&spans, map, &mut tally);
                let mut flat = Blocks::new(std::slice::from_ref(layer), arm.n_host)?;
                let mut ccd = Blocks::new(std::slice::from_ref(layer), arm.n_host)?;
                let xc = &bench.xs[x];
                engine_layer(Lanes::Flat, layer, slots, xc, &mut flat, &mut tally)?;
                engine_layer(Lanes::Ccd(map), layer, slots, xc, &mut ccd, &mut tally)?;
                let n = arm.n_host;
                let eq = |f: &[Tensor2], c: &[Tensor2]| {
                    f.iter().zip(c).all(|(f, c)| bits_equal(&f.data, &c.data))
                };
                let same = eq(&flat.gu[..2 * n], &ccd.gu[..2 * n])
                    && eq(&flat.down[..n], &ccd.down[..n])
                    && eq(&flat.par[..n], &ccd.par[..n]);
                (reached, same)
            } else {
                let host = bench
                    .hosts
                    .get(l)
                    .ok_or("no host layer for the union arm")?;
                let ff = layer.weight(0, GATE).n();
                let mut u = UnionBlocks::new(&bench.xs, arm.rows, embd, ff, bench.n_used)?;
                fill_lists(&mut u.lists, layer, slots, arm.n_host)?;
                let warm_ids = first_distinct_ids(&u.lists, arm.n_host, arm.warm);
                let mut spans = Vec::new();
                host.warm_spans(R8Source::rows(bench.split), &warm_ids, &mut spans)?;
                let reached = warm_job(&spans, map, &mut tally);
                let mut flat = UnionBlocks::new(&bench.xs, arm.rows, embd, ff, bench.n_used)?;
                let call = (arm.n_host, arm.union);
                union5_layer(
                    host,
                    bench.split,
                    layer,
                    slots,
                    call.0,
                    call.1,
                    &mut flat,
                    x,
                    &mut tally,
                )?;
                union5_warm_layer(
                    map,
                    host,
                    bench.split,
                    layer,
                    slots,
                    (arm.n_host, arm.union, 0),
                    &mut u,
                    x,
                    &mut tally,
                )?;
                (reached, bits_equal(&flat.out, &u.out))
            };
            if reached != want {
                short.push(format!("{} ({reached} of {want})", layer.index));
            }
            if !same {
                differ.push(layer.index);
            }
        }
        println!(
            "check arm={} layers={} warm_bytes_equal_k_experts={} outputs_bits_equal_flat={}",
            arm.label(),
            layers.len(),
            short.is_empty(),
            differ.is_empty()
        );
        if !short.is_empty() {
            return Err(format!(
                "arm {}: the warm reached other than {} experts' bytes on layers {short:?}",
                arm.label(),
                arm.warm
            )
            .into());
        }
        if !differ.is_empty() {
            return Err(format!(
                "arm {}: its CCD-major leg differs from the flat leg on layers {differ:?}",
                arm.label()
            )
            .into());
        }
    }
    Ok(())
}

/// The layers the check covers: 0, 1, 2, the last and every layer whose
/// stacks span more than one shard; for a `qwen4exp` file also the first
/// layer of every routed type triple (gate, up, down) those lack.
fn covered_layers(layers: &[Layer<'_>], family: &Family) -> Vec<usize> {
    let last = layers.len().saturating_sub(1);
    let mut covered: Vec<usize> = layers
        .iter()
        .filter(|l| l.index < 3 || l.index == last || l.spans_shards())
        .map(|l| l.index)
        .collect();
    if let Family::Qwen4exp(_) = family {
        let triple = |l: &Layer<'_>| l.stacks.map(|s| s.info.ty);
        let mut seen: Vec<[GgmlType; 3]> = covered.iter().map(|&i| triple(&layers[i])).collect();
        for l in layers {
            if !seen.contains(&triple(l)) {
                seen.push(triple(l));
                covered.push(l.index);
            }
        }
        covered.sort_unstable();
    }
    covered
}

/// The whole check (see the module doc); `Ok` only when every site passed.
fn check(bench: &Bench<'_>, n_host: usize, verbose: bool) -> Result<(), BenchError> {
    let (layers, xs) = (&bench.layers, &bench.xs);
    let covered = covered_layers(layers, bench.family);
    let mut c = Checks {
        verbose,
        sites: 0,
        failed: Vec::new(),
        worst: 0.0,
    };
    check_decoders(&mut c, layers, &covered)?;
    for &li in &covered {
        let slots = distinct(&mut Rng::new(&[KEY_CHECK, li as u64]), n_host, WORKING_SET);
        check_layer(&mut c, &layers[li], &slots, &xs[li % N_X])?;
        let host = bench
            .hosts
            .get(li)
            .ok_or("no host layer to check the union shape")?;
        check_union(
            &mut c,
            &layers[li],
            host,
            bench.split,
            &slots,
            xs,
            bench.n_used,
        )?;
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
    /// One union call per layer over every row in the chunked flow
    /// ([`union_chunks_call`]): each distinct expert read once, every
    /// column's weighted sum at the end.
    Union,
    /// The engine's union call ([`HostLayer::experts_union_into`]): the same
    /// pass in five pool dispatches a layer.
    Union5,
    /// The union call with gate and up through the row-lane Q3_K tile on
    /// the repacked working set.
    UnionR8,
    /// The `unionr8` call with the column tile on the file's rows: its
    /// dispatch, the union shape's kernel.
    UnionQ,
    /// The `union5` pass over qdot's card-rule kernels (the card's own
    /// bits on the host): x quantized once per column, each distinct
    /// expert's gate·up·SwiGLU, the h quantizer, its down, the weighted
    /// sums — five pool dispatches a layer.
    UnionCard,
    /// One dispatch per expert matrix.
    PerMatrix,
    /// The `engine` shape on CCD-major lanes ([`Lanes::Ccd`]) after the engine's
    /// warm routine ([`warm_share`]) filled the first `w<k>` of its experts
    /// into the L3s of the CCDs that compute them: the one-column leg of a
    /// layer whose predicted experts were warmed. The warm is its own pool job,
    /// tallied apart from the two dispatches of the leg.
    EngineWarm,
    /// The `union5` call on CCD-major lanes after the same warm of the first
    /// `w<k>` of its distinct experts ([`HostLayer::warm_spans`]).
    Union5Warm,
    /// The engine shape's dispatches reading the file mapping, no kernel.
    ReadMmap,
    /// The same reads from the anonymous huge-page copy ([`ThpCopy`]).
    ReadThp,
}

impl Shape {
    const ALL: [Shape; 12] = [
        Shape::Engine,
        Shape::EngineSep,
        Shape::Union,
        Shape::Union5,
        Shape::UnionR8,
        Shape::UnionQ,
        Shape::UnionCard,
        Shape::PerMatrix,
        Shape::EngineWarm,
        Shape::Union5Warm,
        Shape::ReadMmap,
        Shape::ReadThp,
    ];

    fn name(self) -> &'static str {
        match self {
            Shape::Engine => "engine",
            Shape::EngineSep => "engine-sep",
            Shape::Union => "union",
            Shape::Union5 => "union5",
            Shape::UnionR8 => "unionr8",
            Shape::UnionQ => "unionq",
            Shape::UnionCard => "unioncard",
            Shape::PerMatrix => "per-matrix",
            Shape::EngineWarm => "engine-warm",
            Shape::Union5Warm => "union5-warm",
            Shape::ReadMmap => "read-mmap",
            Shape::ReadThp => "read-thp",
        }
    }

    /// The shape runs one row per step only.
    fn one_row(self) -> bool {
        matches!(
            self,
            Shape::PerMatrix | Shape::EngineWarm | Shape::ReadMmap | Shape::ReadThp
        )
    }

    /// The shape warms the first `w<k>` of its experts before its leg.
    fn warms(self) -> bool {
        matches!(self, Shape::EngineWarm | Shape::Union5Warm)
    }

    /// One union call per layer: each distinct expert read once.
    fn is_union(self) -> bool {
        matches!(
            self,
            Shape::Union
                | Shape::Union5
                | Shape::Union5Warm
                | Shape::UnionR8
                | Shape::UnionQ
                | Shape::UnionCard
        )
    }

    /// The gate/up dispatch hands out claimed units: the `b<B>` lever.
    fn claims_units(self) -> bool {
        matches!(self, Shape::UnionR8 | Shape::UnionQ)
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
    /// Units a claim of the gate/up dispatch takes (`b<B>`, the `unionr8` and
    /// `unionq` shapes; 1 otherwise).
    claim_block: usize,
    /// Experts the warm shapes warm before the leg (`w<k>`, at most the arm's
    /// distinct experts; 0 otherwise).
    warm: usize,
}

impl Arm {
    /// One arm of `--arms`; `union` is `--union`'s ratio, taken by a
    /// multi-row arm that names none of its own.
    fn parse(s: &str, union: Option<f64>) -> Result<Arm, String> {
        let (name, n) = s.split_once(':').ok_or_else(|| {
            format!(
                "arm {s:?}: want <engine|engine-sep|union|union5|unionr8|unionq|per-matrix|engine-warm|union5-warm|read-mmap|read-thp>:<n_host>[x<rows>[u<r>]][b<B>][w<k>]"
            )
        })?;
        let shape = Shape::ALL
            .into_iter()
            .find(|sh| sh.name() == name)
            .ok_or_else(|| format!("arm {s:?}: unknown shape {name:?}"))?;
        let (n, warm) = match n.split_once('w') {
            Some((n, k)) => {
                if !shape.warms() {
                    return Err(format!(
                        "arm {s:?}: w<k> is the engine-warm and union5-warm warm count"
                    ));
                }
                let k = k
                    .parse::<usize>()
                    .map_err(|_| format!("arm {s:?}: warm count {k:?} is not a count"))?;
                (n, Some(k))
            }
            None => (n, None),
        };
        if shape.warms() && warm.is_none() {
            return Err(format!(
                "arm {s:?}: the {name} shape names how many experts it warms: ...w<k>"
            ));
        }
        let warm = warm.unwrap_or(0);
        let (n, claim_block) = match n.split_once('b') {
            Some((n, b)) => {
                let b = b.parse::<usize>().ok().filter(|&b| b > 0).ok_or_else(|| {
                    format!("arm {s:?}: claim block {b:?} is not a positive count")
                })?;
                if !shape.claims_units() {
                    return Err(format!(
                        "arm {s:?}: b<B> is the unionr8 and unionq gate/up claim block"
                    ));
                }
                (n, b)
            }
            None => (n, 1),
        };
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
        if shape.is_union() && rows > UNION_MAX_COLS {
            return Err(format!(
                "arm {s:?}: a union call takes at most {UNION_MAX_COLS} rows"
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
        if warm > union {
            return Err(format!(
                "arm {s:?}: it warms {warm} of its {union} distinct experts"
            ));
        }
        Ok(Arm {
            shape,
            n_host,
            rows,
            ratio,
            union,
            claim_block,
            warm,
        })
    }

    fn label(self) -> String {
        let shape = self.shape.name();
        let block = if self.claim_block == 1 {
            String::new()
        } else {
            format!("b{}", self.claim_block)
        };
        let warm = if self.shape.warms() {
            format!("w{}", self.warm)
        } else {
            String::new()
        };
        match (self.rows, self.ratio) {
            (1, _) => format!("{shape}:{}{block}{warm}", self.n_host),
            (rows, None) => format!("{shape}:{}x{rows}{block}{warm}", self.n_host),
            (rows, Some(r)) => format!("{shape}:{}x{rows}u{r}{block}{warm}", self.n_host),
        }
    }

    /// Columns each distinct expert of a layer serves: the arm's slots over
    /// its distinct experts.
    fn cols_per_expert(self) -> f64 {
        (self.n_host * self.rows) as f64 / self.union as f64
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
            .filter(|l| self.runs(l))
            .map(|l| self.union as u64 * l.expert_bytes())
            .sum()
    }

    /// Whether this arm's pass runs layer `l`: every arm runs every layer
    /// but `unioncard`, which runs the q4_K-down layers only (the card rule's
    /// down) — its per-token sums cover those layers alone.
    fn runs(self, l: &Layer<'_>) -> bool {
        !matches!(self.shape, Shape::UnionCard) || l.stacks[DOWN].info.ty == GgmlType::Q4_K
    }

    /// Pool dispatches one token of this arm issues, by construction — for
    /// the chunked union shapes their gate/up and down dispatches, a floor
    /// (their fill, pre-pass, copy-aside and sum dispatches follow the plan).
    fn dispatches_per_token(self, layers: &[Layer<'_>]) -> usize {
        match self.shape {
            Shape::PerMatrix => 3 * self.n_host * layers.len(),
            Shape::EngineSep => ENGINE_DISPATCHES * self.rows * layers.len(),
            Shape::Union5 if self.rows > DEFER_MAX_COLS => 5 * layers.len(),
            Shape::Union5 => ENGINE_DISPATCHES * layers.len(),
            // The call's own dispatches and, when it warms, the warm's one.
            Shape::Union5Warm => {
                let call = if self.rows > DEFER_MAX_COLS {
                    5
                } else {
                    ENGINE_DISPATCHES
                };
                (call + usize::from(self.warm > 0)) * layers.len()
            }
            Shape::EngineWarm => (ENGINE_DISPATCHES + usize::from(self.warm > 0)) * layers.len(),
            Shape::UnionCard => 5 * layers.iter().filter(|l| self.runs(l)).count(),
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
    /// `--check`, and the arms whose pre-timing check runs on every layer.
    Check(Vec<Arm>),
    Time(TimeOpts),
}

fn parse_args(args: &[String]) -> Result<Mode, String> {
    let mut it = args.iter().map(String::as_str);
    match it.next() {
        Some("--check") => match (it.next(), it.next(), it.next()) {
            (None, _, _) => Ok(Mode::Check(Vec::new())),
            (Some("--arms"), Some(arms), None) => Ok(Mode::Check(
                arms.split(',')
                    .map(|a| Arm::parse(a, None))
                    .collect::<Result<_, _>>()?,
            )),
            (Some(extra), ..) => Err(format!(
                "--check takes nothing or --arms A,B,..., got {extra:?}"
            )),
        },
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
    family: &'a Family,
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
    /// The file's routed width: every union scratch is made for it, as the
    /// engine's is.
    n_used: usize,
    /// The pool's CCD map, present once every participant is pinned; the warm
    /// shapes need it.
    ccd: Option<CcdMap>,
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
            Shape::UnionQ => Ok(GateUpTile::Column),
            other => Err(format!("the {} shape has no gate-and-up tile", other.name()).into()),
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
                    engine_layer(Lanes::Flat, layer, slots, xs()[0], b, tally)?;
                }
                Shape::EngineSep => {
                    for (row, x) in slots.chunks_exact(arm.n_host).zip(&xs()) {
                        engine_layer(Lanes::Flat, layer, row, x, b, tally)?;
                    }
                }
                Shape::Engine => engine_rows_layer(Lanes::Flat, layer, slots, &xs(), b, tally)?,
                Shape::EngineWarm => {
                    let map = self.ccd.ok_or(NO_CCD_MAP)?;
                    engine_warm_layer(map, layer, slots, arm.warm, xs()[0], b, tally)?;
                }
                Shape::Union5Warm => {
                    let map = self.ccd.ok_or(NO_CCD_MAP)?;
                    let u = b
                        .union
                        .as_mut()
                        .ok_or("union5-warm arm without its blocks")?;
                    let host = self
                        .hosts
                        .get(l)
                        .ok_or("union5-warm arm without host layers")?;
                    let x = (t + l) % N_X;
                    let call = (arm.n_host, arm.union, arm.warm);
                    union5_warm_layer(map, host, self.split, layer, slots, call, u, x, tally)?;
                }
                Shape::Union => {
                    let u = b.chunks.as_mut().ok_or("union arm without its blocks")?;
                    let host = self.hosts.get(l).ok_or("union arm without host layers")?;
                    let x = (t + l) % N_X;
                    union_layer(
                        host, self.split, layer, slots, arm.n_host, arm.union, u, x, tally,
                    )?;
                }
                Shape::Union5 => {
                    let u = b.union.as_mut().ok_or("union5 arm without its blocks")?;
                    let host = self.hosts.get(l).ok_or("union5 arm without host layers")?;
                    let x = (t + l) % N_X;
                    union5_layer(
                        host, self.split, layer, slots, arm.n_host, arm.union, u, x, tally,
                    )?;
                }
                Shape::UnionCard => {
                    let c = b.card.as_mut().ok_or("unioncard arm without its blocks")?;
                    let x = (t + l) % N_X;
                    unioncard_layer(layer, slots, arm.n_host, c, x, tally)?;
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
            match arm.shape {
                Shape::Union => {
                    blocks.chunks =
                        Some(ChunkBlocks::new(&self.xs, arm.rows, embd, ff, self.n_used))
                }
                Shape::Union5 | Shape::Union5Warm => {
                    blocks.union =
                        Some(UnionBlocks::new(&self.xs, arm.rows, embd, ff, self.n_used)?);
                }
                Shape::UnionCard => {
                    blocks.card = Some(CardBlocks::new(&self.xs, arm.rows, embd, ff, arm.n_host));
                }
                // `unionr8` and `unionq` share the blocks.
                _ => {
                    blocks.r8 = Some(R8Blocks::new(
                        &self.xs,
                        (arm.rows, embd, ff),
                        self.n_used,
                        arm.union,
                        arm.claim_block,
                    ));
                }
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

/// A warm arm's two numbers from the dispatch rows of its rounds, as `key=value`
/// fields: the warm job's rate in GB/s (the bytes its walks reached over its
/// wall) and the leg's cost per expert in µs (the leg's dispatches' wall over
/// the experts they ran, the warm left out). Empty for an arm that does not warm.
fn warm_fields<'r>(arm: &Arm, rows: impl Iterator<Item = &'r TallyRow>) -> String {
    if !arm.shape.warms() {
        return String::new();
    }
    let (mut warm_bytes, mut warm_ns, mut leg_ns, mut calls) = (0u64, 0u64, 0u64, 0u64);
    for r in rows {
        match r.kind {
            "warm" => {
                warm_bytes += r.bytes * r.count;
                warm_ns += r.ns;
            }
            "gate+up" | "union5" => {
                leg_ns += r.ns;
                calls += r.count;
            }
            "down" => leg_ns += r.ns,
            _ => {}
        }
    }
    let experts = if arm.shape == Shape::Union5Warm {
        arm.union
    } else {
        arm.n_host
    };
    let rate = warm_bytes as f64 / warm_ns.max(1) as f64;
    let leg = leg_ns as f64 / calls.max(1) as f64 / experts as f64 / 1e3;
    format!(
        " warm_k={} warm_gbps={rate:.2} leg_us_per_expert={leg:.2}",
        arm.warm
    )
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
                 minflt={} majflt={} resident_before={}/{total} resident_after={}/{total} lease={} admissible={}{}",
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
                if ok { "yes" } else { "no" },
                warm_fields(&arm, r.tally.rows.iter())
            );
            for d in &r.tally.rows {
                let us = d.ns as f64 / d.count as f64 / 1e3;
                let per_expert = if arm.shape.is_union() {
                    format!(
                        " cols_per_expert={:.2} us_per_expert={:.2}",
                        arm.cols_per_expert(),
                        us / arm.union as f64
                    )
                } else {
                    String::new()
                };
                println!(
                    "dispatch threads={threads} round={}/{} arm={} kind={} bytes={} count={} us_mean={us:.2} gbps={:.2}{per_expert}",
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
        let warm = warm_fields(arm, rs.iter().flat_map(|r| r.tally.rows.iter()));
        let per_expert = if arm.shape.is_union() {
            let calls = (bench.layers.len() * arm.union) as f64;
            format!(
                " cols_per_expert={:.2} us_per_expert={:.2} us_per_expert_min={:.2}",
                arm.cols_per_expert(),
                mean * 1e3 / calls,
                min * 1e3 / calls
            )
        } else {
            String::new()
        };
        println!(
            "summary threads={threads} arm={} rounds={} tokens={} ms_min={min:.3} ms_mean={mean:.3} round_means=[{}] \
             gbps_mean={:.2} gbps_best={:.2} admissible={}{per_expert}{thp}{warm}",
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
/// plain combine the other shapes run); a `qwen4exp` file's as
/// `arch::qwen35moe::host::layers` builds them (its routed SwiGLU has no
/// limit), every refused layer named.
fn host_layers(
    split: &Split,
    meta: &Meta,
    family: &Family,
    layers: &[Layer<'_>],
) -> Result<Vec<HostLayer>, BenchError> {
    if let Family::Qwen4exp(hp) = family {
        return Ok(qwen4exp_host::layers(
            R8Source::rows(split),
            hp,
            0..layers.len(),
        )?);
    }
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
            Ok(HostLayer::build(R8Source::rows(split), &spec)?)
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
            "BLOOMERY_REF_MODEL unset — run through tools/box.sh with BLOOMERY_MODEL=deepseek41 \
             or qwen4exp, or the just recipes"
                .into()
        })
}

fn run(mode: Mode) -> Result<(), BenchError> {
    let path = model_path()?;
    let split = Split::open(&path)?;
    let family = Family::of(&split)?;
    let meta = Meta::read(&split, &family)?;
    let layers = build_layers(&split, &meta, &family)?;
    print_table(&path, &split, &meta, &family, &layers);
    let arms: &[Arm] = match &mode {
        Mode::Time(opts) => &opts.arms,
        Mode::Check(arms) => arms,
    };
    let past_used = arms.iter().find(|a| a.n_host > meta.n_used).copied();
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
    let ccd = threads::pool().ccd_map();
    match ccd {
        Some(m) => println!(
            "v41host ccd_map threads={} ccds={} widths={:?} l3_bytes={}",
            m.threads(),
            m.ccds(),
            (0..m.ccds()).map(|c| m.width(c)).collect::<Vec<_>>(),
            m.l3_bytes()
        ),
        None => println!(
            "v41host ccd_map=off pin_failed={} caller_pinned={}",
            threads::pool().pin_failed(),
            threads::pool().caller_pinned()
        ),
    }
    // A plain `--check` runs the warm arms too: with no CCD map it is refused
    // by name before the load, as an explicit warm arm is, not passed green
    // with the clause unrun.
    let plain_check = matches!(&mode, Mode::Check(arms) if arms.is_empty());
    if ccd.is_none() && (plain_check || arms.iter().any(|a| a.shape.warms())) {
        return Err(NO_CCD_MAP.into());
    }
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
    let r8_layers = if arms.iter().any(|a| a.shape == Shape::UnionR8) {
        (0..layers.len()).collect()
    } else if matches!(mode, Mode::Check(_)) && matches!(family, Family::Discovered) {
        covered_layers(&layers, &family)
    } else {
        Vec::new()
    };
    if matches!(mode, Mode::Check(_)) && !matches!(family, Family::Discovered) {
        println!(
            "v41host unionr8_unionq_check=none family={}: its routed gates and ups are not Q3_K, the \
             row-lane and column tiles' type",
            family.name()
        );
    }
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
    let hosts = host_layers(&split, &meta, &family, &layers)?;
    let bench = Bench {
        split: &split,
        family: &family,
        layers,
        hosts,
        r8,
        xs: activations(meta.embd),
        pages,
        mapped,
        thp: copy.as_ref().map(ThpCopy::bytes),
        sink: AtomicU64::new(0),
        n_used: meta.n_used,
        ccd,
    };
    match mode {
        Mode::Check(arms) if arms.is_empty() => {
            check(&bench, meta.n_used, true)?;
            let default_arms = match &family {
                Family::Discovered => R8_CHECK_ARMS,
                Family::Qwen4exp(_) => Q38_CHECK_ARMS,
            };
            let arms = default_arms
                .split(',')
                .map(|a| Arm::parse(a, None))
                .collect::<Result<Vec<_>, _>>()?;
            check_r8_arms(&bench, &arms, &covered_layers(&bench.layers, &family))?;
            let warm_arms = WARM_CHECK_ARMS
                .split(',')
                .map(|a| Arm::parse(a, None))
                .collect::<Result<Vec<_>, _>>()?;
            check_warm_arms(&bench, &warm_arms, &covered_layers(&bench.layers, &family))?;
            let card_arms = CARD_CHECK_ARMS
                .split(',')
                .map(|a| Arm::parse(a, None))
                .collect::<Result<Vec<_>, _>>()?;
            if matches!(family, Family::Discovered) {
                check_card_arms(&bench, &card_arms, &covered_layers(&bench.layers, &family))
            } else {
                println!(
                    "check unioncard arms skipped: {CARD_CHECK_ARMS} take q3_K gates and ups, \
                     this file's routed stacks are not them"
                );
                Ok(())
            }
        }
        Mode::Check(arms) => {
            check(&bench, meta.n_used, true)?;
            let every: Vec<usize> = (0..bench.layers.len()).collect();
            check_r8_arms(&bench, &arms, &every)?;
            check_warm_arms(&bench, &arms, &every)?;
            check_card_arms(&bench, &arms, &every)
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
            check_warm_arms(&bench, &opts.arms, &every)?;
            check_card_arms(&bench, &opts.arms, &every)?;
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
