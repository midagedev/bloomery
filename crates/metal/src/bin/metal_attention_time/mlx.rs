//! MLX's decode attention, timed beside ours on the same cache rows: the
//! reference rate the flash is held to. The kernels are MLX's
//! `sdpa_vector`, `sdpa_vector_2pass_1` and `sdpa_vector_2pass_2`, compiled
//! after the shipped source in the probe's library, and the block count is
//! MLX's own rule for the GPU class this probe runs on. A development-loop
//! instrument: its figures are this machine's.

use bloomery_device::conformance::why;
use bloomery_device::{Buffer, Device, DeviceError, Elems};
use bloomery_metal::MetalStream;
use bloomery_metal::buffer::MetalBuffer;
use bloomery_metal::device::MetalDevice;
use bloomery_metal::encode::{Binding, ComputeEncoder};
use bloomery_metal::kernels::{KV_HEADS, QUERY_HEADS};
use bloomery_metal::library::Pipeline;

/// MLX's kernels at this build's head, ported to read the engine's buffers.
pub const SOURCE: &str = r#"
// MLX's decode SDPA kernels, for timing beside the engine's flash. Ported
// from ml-explore/mlx v0.32.3, mlx/backend/metal/kernels/sdpa_vector.h
// (MIT licence, Copyright 2024 Apple Inc.): sdpa_vector, sdpa_vector_2pass_1
// and sdpa_vector_2pass_2 at T = half, D = V = BLOOMERY_ATTN_HEAD, one query,
// no mask, no sinks. The arithmetic and the walk are MLX's; what differs is
// where the engine's buffers differ: the query is the engine's f32 rows, the
// cache is the engine's rows (row kv * ctx + key, HEAD halves a row: MLX's
// dense [kv][key][dim] with k_head_stride ctx * HEAD and k_seq_stride HEAD),
// the outputs are f32, MLX's 3-D grid is this encoder's 2-D one, and pass 1's
// block count is a launch word (MLX: a function constant). MLX builds its
// kernels without fast math and lets the compiler contract.
#pragma METAL fp contract(on)

#define MLX_D BLOOMERY_ATTN_HEAD

struct MlxArgs {
    uint n;
    uint ctx;
    float scale;
    uint blocks;
};

// sdpa_vector: a threadgroup a query head, 32 SIMD groups striding the keys.
kernel void mlx_vec_f16(device float const *queries [[buffer(0)]],
                        device half const *keys [[buffer(1)]],
                        device half const *values [[buffer(2)]],
                        device float *out [[buffer(3)]],
                        constant MlxArgs &args [[buffer(4)]],
                        uint3 tid [[threadgroup_position_in_grid]],
                        uint simd_gid [[simdgroup_index_in_threadgroup]],
                        uint simd_lid [[thread_index_in_simdgroup]]) {
    constexpr int BN = 32;
    constexpr int BD = 32;
    constexpr int qk_per_thread = MLX_D / BD;
    constexpr int v_per_thread = MLX_D / BD;
    const int N = int(args.n);
    const size_t k_seq_stride = MLX_D;
    const size_t k_head_stride = size_t(args.ctx) * MLX_D;
    int inner_k_stride = BN * int(k_seq_stride);
    int inner_v_stride = BN * int(k_seq_stride);

    typedef float U;

    thread U q[qk_per_thread];
    thread U k[qk_per_thread];
    thread U o[v_per_thread];

    threadgroup U outputs[BN * BD];
    threadgroup U max_scores[BN];
    threadgroup U sum_exp_scores[BN];

    const int q_batch_head_idx = tid.x;
    const int kv_head_idx = q_batch_head_idx / BLOOMERY_ATTN_GQA;
    queries += q_batch_head_idx * BLOOMERY_ATTN_QG + simd_lid * qk_per_thread;
    keys += kv_head_idx * k_head_stride + simd_gid * k_seq_stride + simd_lid * qk_per_thread;
    values += kv_head_idx * k_head_stride + simd_gid * k_seq_stride + simd_lid * v_per_thread;
    out += q_batch_head_idx * MLX_D + simd_gid * v_per_thread;

    for (int i = 0; i < qk_per_thread; i++) {
        q[i] = args.scale * queries[i];
    }
    for (int i = 0; i < v_per_thread; i++) {
        o[i] = 0;
    }

    U max_score = -FLT_MAX;
    U sum_exp_score = 0;

    for (int i = simd_gid; i < N; i += BN) {
        for (int j = 0; j < qk_per_thread; j++) {
            k[j] = keys[j];
        }
        U score = 0;
        for (int j = 0; j < qk_per_thread; j++) {
            score += q[j] * k[j];
        }
        score = simd_sum(score);
        U new_max = max(max_score, score);
        U factor = fast::exp(max_score - new_max);
        U exp_score = fast::exp(score - new_max);
        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;
        for (int j = 0; j < v_per_thread; j++) {
            o[j] = o[j] * factor + exp_score * values[j];
        }
        keys += inner_k_stride;
        values += inner_v_stride;
    }

    if (simd_lid == 0) {
        max_scores[simd_gid] = max_score;
        sum_exp_scores[simd_gid] = sum_exp_score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    max_score = max_scores[simd_lid];
    U new_max = simd_max(max_score);
    U factor = fast::exp(max_score - new_max);
    sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);

    for (int i = 0; i < v_per_thread; i++) {
        outputs[simd_lid * BD + simd_gid] = o[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
        o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (simd_lid == 0) {
        for (int i = 0; i < v_per_thread; i++) {
            out[i] = o[i];
        }
    }
}

// sdpa_vector_2pass_1: a threadgroup a (key head, block) of the key head's
// query heads, a SIMD group each; block b holds keys b, b + blocks, ...
kernel void mlx_2p1_f16(device float const *queries [[buffer(0)]],
                        device half const *keys [[buffer(1)]],
                        device half const *values [[buffer(2)]],
                        device half *out [[buffer(3)]],
                        device float *sums [[buffer(4)]],
                        device float *maxs [[buffer(5)]],
                        constant MlxArgs &args [[buffer(6)]],
                        uint3 tid [[threadgroup_position_in_grid]],
                        uint simd_gid [[simdgroup_index_in_threadgroup]],
                        uint simd_lid [[thread_index_in_simdgroup]]) {
    constexpr int BD = 32;
    constexpr int qk_per_thread = MLX_D / BD;
    constexpr int v_per_thread = MLX_D / BD;

    typedef float U;

    thread U q[qk_per_thread];
    thread U o[v_per_thread] = {0};

    const int N = int(args.n);
    const int blocks = int(args.blocks);
    const size_t k_seq_stride = MLX_D;
    const size_t k_head_stride = size_t(args.ctx) * MLX_D;
    const int kv_head_idx = tid.x;
    const int block_idx = tid.y;
    const int q_head_idx = BLOOMERY_ATTN_GQA * kv_head_idx + int(simd_gid);
    const int o_offset = q_head_idx;

    queries += q_head_idx * BLOOMERY_ATTN_QG + simd_lid * qk_per_thread;
    keys += kv_head_idx * k_head_stride + block_idx * k_seq_stride + simd_lid * qk_per_thread;
    values += kv_head_idx * k_head_stride + block_idx * k_seq_stride + simd_lid * v_per_thread;
    out += o_offset * blocks * MLX_D + block_idx * MLX_D + simd_lid * v_per_thread;
    sums += o_offset * blocks + block_idx;
    maxs += o_offset * blocks + block_idx;

    for (int i = 0; i < qk_per_thread; i++) {
        q[i] = args.scale * queries[i];
    }

    U max_score = -FLT_MAX;
    U sum_exp_score = 0;

    for (int i = block_idx; i < N; i += blocks) {
        U score = 0;
        for (int j = 0; j < qk_per_thread; j++) {
            score += q[j] * keys[j];
        }
        score = simd_sum(score);
        U new_max = max(max_score, score);
        U factor = fast::exp(max_score - new_max);
        U exp_score = fast::exp(score - new_max);
        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;
        for (int j = 0; j < v_per_thread; j++) {
            o[j] = o[j] * factor + exp_score * values[j];
        }
        keys += blocks * int(k_seq_stride);
        values += blocks * int(k_seq_stride);
    }

    if (simd_lid == 0) {
        sums[0] = sum_exp_score;
        maxs[0] = max_score;
    }
    for (int i = 0; i < v_per_thread; i++) {
        out[i] = static_cast<half>(o[i]);
    }
}

// The step-2 arms: pass 1 with one change each, writing pass 1's partials so
// pass 2 folds them. CONTIG walks a block's keys as one run (the engine's
// segments) instead of every blocks-th key; EXACT takes precise exp for
// fast::exp; CHUNK is the engine's walk (lane words of four keys in flight,
// the score scaled after the sum, one rescale a chunk, precise exp, fma).
// `pad` is threadgroup memory the kernel declares and holds live through a
// branch no launch takes: its size is the arm.
template <bool CONTIG, bool EXACT, bool CHUNK>
static void mlx_p1_arm(device float const *queries,
                       device half const *keys,
                       device half const *values,
                       device half *out,
                       device float *sums,
                       device float *maxs,
                       constant MlxArgs &args,
                       uint3 tid,
                       uint simd_gid,
                       uint simd_lid,
                       threadgroup float *pad,
                       uint pad_n) {
    constexpr int BD = 32;
    constexpr int per = MLX_D / BD;
    typedef float U;
    thread U q[per];
    thread U o[per] = {0};

    const int N = int(args.n);
    const int blocks = int(args.blocks);
    const size_t k_seq_stride = MLX_D;
    const size_t k_head_stride = size_t(args.ctx) * MLX_D;
    const int kv_head_idx = tid.x;
    const int block_idx = tid.y;
    const int q_head_idx = BLOOMERY_ATTN_GQA * kv_head_idx + int(simd_gid);

    queries += q_head_idx * BLOOMERY_ATTN_QG + simd_lid * per;
    keys += kv_head_idx * k_head_stride + simd_lid * per;
    values += kv_head_idx * k_head_stride + simd_lid * per;
    out += q_head_idx * blocks * MLX_D + block_idx * MLX_D + simd_lid * per;
    sums += q_head_idx * blocks + block_idx;
    maxs += q_head_idx * blocks + block_idx;

    int lo = block_idx;
    int hi = N;
    int step = blocks;
    if (CONTIG || CHUNK) {
        int span = (N + blocks - 1) / blocks;
        lo = min(block_idx * span, N);
        hi = min(lo + span, N);
        step = 1;
    }

    U max_score = -FLT_MAX;
    U sum_exp_score = 0;
    if (CHUNK) {
        for (int i = 0; i < per; i++) {
            q[i] = queries[i];
        }
        max_score = -INFINITY;
        for (int base = lo; base < hi; base += 4) {
            half kw[4][per];
            half vw[4][per];
            for (int c = 0; c < 4; ++c) {
                int j = min(base + c, hi - 1);
                for (int d = 0; d < per; ++d) {
                    kw[c][d] = keys[size_t(j) * k_seq_stride + d];
                    vw[c][d] = values[size_t(j) * k_seq_stride + d];
                }
            }
            U s[4];
            U chunk_max = -INFINITY;
            for (int c = 0; c < 4; ++c) {
                U dot = 0.0f;
                for (int d = 0; d < per; ++d) {
                    dot = fma(q[d], U(kw[c][d]), dot);
                }
                dot = simd_sum(dot);
                U score = dot * args.scale;
                s[c] = base + c < hi ? score : -INFINITY;
                chunk_max = max(chunk_max, s[c]);
            }
            U new_m = max(max_score, chunk_max);
            U factor = max_score > -INFINITY ? precise::exp(max_score - new_m) : 0.0f;
            sum_exp_score *= factor;
            for (int d = 0; d < per; ++d) {
                o[d] *= factor;
            }
            max_score = new_m;
            for (int c = 0; c < 4; ++c) {
                U weight = base + c < hi ? precise::exp(s[c] - new_m) : 0.0f;
                sum_exp_score += weight;
                for (int d = 0; d < per; ++d) {
                    o[d] = fma(weight, U(vw[c][d]), o[d]);
                }
            }
        }
    } else {
        for (int i = 0; i < per; i++) {
            q[i] = args.scale * queries[i];
        }
        keys += size_t(lo) * k_seq_stride;
        values += size_t(lo) * k_seq_stride;
        for (int i = lo; i < hi; i += step) {
            U score = 0;
            for (int j = 0; j < per; j++) {
                score += q[j] * keys[j];
            }
            score = simd_sum(score);
            U new_max = max(max_score, score);
            U factor = EXACT ? precise::exp(max_score - new_max) : fast::exp(max_score - new_max);
            U exp_score = EXACT ? precise::exp(score - new_max) : fast::exp(score - new_max);
            max_score = new_max;
            sum_exp_score = sum_exp_score * factor + exp_score;
            for (int j = 0; j < per; j++) {
                o[j] = o[j] * factor + exp_score * values[j];
            }
            keys += step * int(k_seq_stride);
            values += step * int(k_seq_stride);
        }
    }

    if (simd_gid * BD + simd_lid < pad_n) {
        pad[simd_gid * BD + simd_lid] = max_score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (args.blocks > 0x7fffffffu) {
        sum_exp_score += pad[(simd_lid * 7u + 1u) % pad_n];
    }
    if (simd_lid == 0) {
        sums[0] = sum_exp_score;
        maxs[0] = max_score;
    }
    for (int i = 0; i < per; i++) {
        out[i] = static_cast<half>(o[i]);
    }
}

#define MLX_ARM(NAME, CONTIG, EXACT, CHUNK, PAD_FLOATS)                                 \
    kernel void NAME(device float const *queries [[buffer(0)]],                         \
                     device half const *keys [[buffer(1)]],                             \
                     device half const *values [[buffer(2)]],                           \
                     device half *out [[buffer(3)]],                                    \
                     device float *sums [[buffer(4)]],                                  \
                     device float *maxs [[buffer(5)]],                                  \
                     constant MlxArgs &args [[buffer(6)]],                              \
                     uint3 tid [[threadgroup_position_in_grid]],                        \
                     uint simd_gid [[simdgroup_index_in_threadgroup]],                  \
                     uint simd_lid [[thread_index_in_simdgroup]]) {                     \
        threadgroup float pad[PAD_FLOATS];                                              \
        mlx_p1_arm<CONTIG, EXACT, CHUNK>(queries, keys, values, out, sums, maxs, args,  \
                                         tid, simd_gid, simd_lid, pad, PAD_FLOATS);     \
    }

MLX_ARM(mlx_arm_base, false, false, false, 1)
MLX_ARM(mlx_arm_pad16400, false, false, false, 4100)
MLX_ARM(mlx_arm_pad16368, false, false, false, 4092)
MLX_ARM(mlx_arm_pad8192, false, false, false, 2048)
MLX_ARM(mlx_arm_exact, false, true, false, 1)
MLX_ARM(mlx_arm_contig, true, false, false, 1)
MLX_ARM(mlx_arm_chunk, true, true, true, 1)
MLX_ARM(mlx_arm_chunk_pad16400, true, true, true, 4100)

// sdpa_vector_2pass_2: a threadgroup a query head of 32 SIMD groups folds
// the blocks' partials, a SIMD group every 32nd block.
kernel void mlx_2p2_f16(device half const *partials [[buffer(0)]],
                        device float const *sums [[buffer(1)]],
                        device float const *maxs [[buffer(2)]],
                        device float *out [[buffer(3)]],
                        constant MlxArgs &args [[buffer(4)]],
                        uint3 tid [[threadgroup_position_in_grid]],
                        uint simd_gid [[simdgroup_index_in_threadgroup]],
                        uint simd_lid [[thread_index_in_simdgroup]]) {
    constexpr int BN = 32;
    constexpr int BD = 32;
    constexpr int elem_per_thread = MLX_D / BD;
    const int blocks = int(args.blocks);

    typedef float U;

    thread U o[elem_per_thread] = {0};
    threadgroup U outputs[BN * BD];

    const int q_offset = tid.x;
    partials += q_offset * blocks * MLX_D + simd_gid * MLX_D + simd_lid * elem_per_thread;
    sums += q_offset * blocks;
    maxs += q_offset * blocks;
    out += q_offset * MLX_D + simd_gid * elem_per_thread;

    U sum_exp_score = 0.0;
    U max_score = -FLT_MAX;

    for (int b = 0; b < blocks / BN; ++b) {
        max_score = max(max_score, maxs[simd_lid + BN * b]);
    }
    max_score = simd_max(max_score);

    for (int b = 0; b < blocks / BN; ++b) {
        U factor = fast::exp(maxs[simd_lid + BN * b] - max_score);
        sum_exp_score += factor * sums[simd_lid + BN * b];
    }
    sum_exp_score = simd_sum(sum_exp_score);

    for (int b = 0; b < blocks / BN; ++b) {
        U factor = fast::exp(maxs[simd_gid] - max_score);
        for (int i = 0; i < elem_per_thread; i++) {
            o[i] += factor * static_cast<U>(partials[i]);
        }
        maxs += BN;
        sums += BN;
        partials += BN * MLX_D;
    }

    for (int i = 0; i < elem_per_thread; i++) {
        outputs[simd_lid * BD + simd_gid] = o[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o[i] = simd_sum(outputs[simd_gid * BD + simd_lid]);
        o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (simd_lid == 0) {
        for (int i = 0; i < elem_per_thread; i++) {
            out[i] = o[i];
        }
    }
}
"#;

/// The most blocks a pass-1 launch takes here: MLX's count at the timer's
/// largest key count.
pub const MAX_BLOCKS: usize = 256;

/// MLX v0.32.3's pass-1 block count on an `applegpu_g13s` GPU
/// (`sdpa_vector_2pass` in scaled_dot_product_attention.cpp): 64, or past
/// 1024 keys with more than four query heads a key head 128 up to 8192 keys
/// and 256 up to 32768.
#[must_use]
pub fn blocks_g13s(keys: u32, gqa: usize) -> usize {
    if keys > 1024 && gqa > 4 {
        match keys {
            ..=8192 => 128,
            _ => 256,
        }
    } else {
        64
    }
}

/// The pass-1 partials, MLX's layout: a query head's blocks of HEAD halves,
/// and a float maximum and weight sum a (head, block).
pub struct MlxBufs {
    pub partials: MetalBuffer<u16>,
    pub sums: MetalBuffer<f32>,
    pub maxs: MetalBuffer<f32>,
    pub out: MetalBuffer<f32>,
}

impl MlxBufs {
    pub fn new(device: &MetalDevice, stream: &MetalStream, head: usize) -> Result<Self, String> {
        let slots = QUERY_HEADS * MAX_BLOCKS;
        Ok(Self {
            partials: device
                .zeroed::<u16>(stream, Elems(slots * head))
                .map_err(why)?,
            sums: device.zeroed::<f32>(stream, Elems(slots)).map_err(why)?,
            maxs: device.zeroed::<f32>(stream, Elems(slots)).map_err(why)?,
            out: device
                .zeroed::<f32>(stream, Elems(QUERY_HEADS * head))
                .map_err(why)?,
        })
    }
}

/// The step-2 arms' kernels: pass 1 with one change each.
pub const ARMS: [&str; 8] = [
    "mlx_arm_base",
    "mlx_arm_pad16400",
    "mlx_arm_pad16368",
    "mlx_arm_pad8192",
    "mlx_arm_exact",
    "mlx_arm_contig",
    "mlx_arm_chunk",
    "mlx_arm_chunk_pad16400",
];

/// The pipelines of the three ported kernels and the arms.
pub struct MlxPipes {
    pub vec: Pipeline,
    pub pass1: Pipeline,
    pub pass2: Pipeline,
    pub arms: Vec<(&'static str, Pipeline)>,
}

/// What one MLX launch reads: the query rows, the cache, the key count and
/// the cache's rows a key head.
pub struct MlxLaunch<'a> {
    pub q: &'a MetalBuffer<f32>,
    pub qg: usize,
    pub cache_k: &'a mut MetalBuffer<u16>,
    pub cache_v: &'a mut MetalBuffer<u16>,
    pub cache_n: usize,
    pub keys: u32,
    pub ctx: u32,
    pub scale: f32,
    pub head: usize,
}

fn args_bytes(l: &MlxLaunch<'_>, blocks: usize) -> Vec<u8> {
    let blocks = u32::try_from(blocks).expect("a block count fits u32");
    [l.keys, l.ctx, l.scale.to_bits(), blocks]
        .iter()
        .flat_map(|w| w.to_ne_bytes())
        .collect()
}

/// `sdpa_vector`: a threadgroup a query head of 1024 threads.
pub fn encode_vec(
    encoder: &mut ComputeEncoder<'_>,
    pipes: &MlxPipes,
    l: &mut MlxLaunch<'_>,
    m: &mut MlxBufs,
) -> Result<(), DeviceError> {
    let args = args_bytes(l, 0);
    encoder.set_pipeline(&pipes.vec);
    encoder.set_binding(0, Binding::of(&l.q.window(Elems(0), Elems(l.qg))?)?)?;
    encoder.set_binding(
        1,
        Binding::of_mut(&l.cache_k.window_mut(Elems(0), Elems(l.cache_n))?)?,
    )?;
    encoder.set_binding(
        2,
        Binding::of_mut(&l.cache_v.window_mut(Elems(0), Elems(l.cache_n))?)?,
    )?;
    let out_n = QUERY_HEADS * l.head;
    encoder.set_binding(
        3,
        Binding::of_mut(&m.out.window_mut(Elems(0), Elems(out_n))?)?,
    )?;
    encoder.set_binding(4, Binding::bytes(&args)?)?;
    encoder.dispatch_groups(QUERY_HEADS, 1, 1024)
}

/// `sdpa_vector_2pass_1` or one of its arms: (key heads, blocks)
/// threadgroups of a SIMD group a query head of the key head.
pub fn encode_pass1(
    encoder: &mut ComputeEncoder<'_>,
    pass1: &Pipeline,
    l: &mut MlxLaunch<'_>,
    m: &mut MlxBufs,
    blocks: usize,
) -> Result<(), DeviceError> {
    let args = args_bytes(l, blocks);
    let slots = QUERY_HEADS * blocks;
    encoder.set_pipeline(pass1);
    encoder.set_binding(0, Binding::of(&l.q.window(Elems(0), Elems(l.qg))?)?)?;
    encoder.set_binding(
        1,
        Binding::of_mut(&l.cache_k.window_mut(Elems(0), Elems(l.cache_n))?)?,
    )?;
    encoder.set_binding(
        2,
        Binding::of_mut(&l.cache_v.window_mut(Elems(0), Elems(l.cache_n))?)?,
    )?;
    encoder.set_binding(
        3,
        Binding::of_mut(&m.partials.window_mut(Elems(0), Elems(slots * l.head))?)?,
    )?;
    encoder.set_binding(
        4,
        Binding::of_mut(&m.sums.window_mut(Elems(0), Elems(slots))?)?,
    )?;
    encoder.set_binding(
        5,
        Binding::of_mut(&m.maxs.window_mut(Elems(0), Elems(slots))?)?,
    )?;
    encoder.set_binding(6, Binding::bytes(&args)?)?;
    encoder.dispatch_groups(KV_HEADS, blocks, 32 * (QUERY_HEADS / KV_HEADS))
}

/// `sdpa_vector_2pass_2`: a threadgroup a query head of 1024 threads.
pub fn encode_pass2(
    encoder: &mut ComputeEncoder<'_>,
    pipes: &MlxPipes,
    l: &MlxLaunch<'_>,
    m: &mut MlxBufs,
    blocks: usize,
) -> Result<(), DeviceError> {
    let args = args_bytes(l, blocks);
    let slots = QUERY_HEADS * blocks;
    encoder.set_pipeline(&pipes.pass2);
    encoder.set_binding(
        0,
        Binding::of_mut(&m.partials.window_mut(Elems(0), Elems(slots * l.head))?)?,
    )?;
    encoder.set_binding(
        1,
        Binding::of_mut(&m.sums.window_mut(Elems(0), Elems(slots))?)?,
    )?;
    encoder.set_binding(
        2,
        Binding::of_mut(&m.maxs.window_mut(Elems(0), Elems(slots))?)?,
    )?;
    let out_n = QUERY_HEADS * l.head;
    encoder.set_binding(
        3,
        Binding::of_mut(&m.out.window_mut(Elems(0), Elems(out_n))?)?,
    )?;
    encoder.set_binding(4, Binding::bytes(&args)?)?;
    encoder.dispatch_groups(QUERY_HEADS, 1, 1024)
}
