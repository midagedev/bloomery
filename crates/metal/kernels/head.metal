// The output head, two dispatches. The first is a quantized mat-vec with the
// norm on the load over the vocabulary's rows (`head_logits_kernel`), the
// instantiations Q6_K, A4, A8, A4H, A8H, Q8_0 and Q5_K under their host names; its row loop
// is matvec_q4k.metal's and it stores every logit. The second
// (`logits_argmax`) is one threadgroup over the stored logits: each thread
// reduces its strided share to one packed key, the threadgroup reduces the
// keys, and thread 0 stores the winning row index. The encoder's barrier
// between the two is what makes the logits visible to the second.
//
// The packed key is ticket.metal's argmax_key, so the lowest row wins an equal
// value. An empty share, and a share whose live logits are all NaN, packs
// (-inf, 0). When that key wins, the step id is not a token.
//
// A non-finite logit raises the logit site. The first dispatch leaves the
// value stored and raises nothing for it: the row loop's ROW_RAISE is off, so
// the norm site is not raised for a row the logit site owns.
//
// This file comes after matvec_q6k.metal (the entry templates) and
// ticket.metal.

#include <metal_stdlib>

using namespace metal;

#pragma METAL fp contract(off)

// The argmax's threadgroup: the mat-vec function constant, the count the
// argmax pipeline is specialized at, so the strides read the same count the
// launch dispatches. A function constant cannot size an array, so the stage is
// sized at the most a threadgroup holds (geometry.rs's
// MAX_SIMD_GROUPS_PER_GROUP: the device's 1024 threads, 32 of them in a
// group) and reduced over the count the launch runs.
#define HEAD_SIMD_GROUPS BLOOMERY_MATVEC_SIMD_GROUPS
#define HEAD_THREADS (32u * HEAD_SIMD_GROUPS)
#define HEAD_STAGE_MAX 32u

// The first dispatch: the logits, the norm on the load. The rows of a
// threadgroup are the mat-vec's own geometry. SCALED: the input is the last
// join's x · gain and `gain` binds that join's scale word (matvec_rows'
// SCALED), so the logits are the norm-on-load entry's bits.
template <typename Q, bool SCALED = false>
kernel void head_logits_kernel(device const typename Q::Block *weights [[buffer(0)]],
                               device const float *x [[buffer(1)]],
                               device const float *gain [[buffer(2)]],
                               device float *logits [[buffer(3)]],
                               device atomic_uint *faults [[buffer(4)]],
                               constant MatvecArgs &args [[buffer(5)]],
                               uint tg [[threadgroup_position_in_grid]],
                               ushort tiisg [[thread_index_in_simdgroup]],
                               ushort sg [[simdgroup_index_in_threadgroup]])
{
    const int first_row = simd_first_row(tg, sg);
    matvec_rows<Q, !SCALED, false, false, false, SCALED>(weights, x, logits, (int)args.nblocks,
                                                         args.eps, args.layer, gain, nullptr,
                                                         faults, first_row, (int)args.rows, tiisg);
}

// Each type's scaled logits entry.
#define SCALED_HEAD_ENTRY(NAME, Q, BQ)                                                 \
    template [[host_name(NAME)]]                                                      \
    kernel void head_logits_kernel<Q, true>(device const BQ *weights [[buffer(0)]],    \
                                            device const float *x [[buffer(1)]],       \
                                            device const float *gain [[buffer(2)]],    \
                                            device float *logits [[buffer(3)]],        \
                                            device atomic_uint *faults [[buffer(4)]],  \
                                            constant MatvecArgs &args [[buffer(5)]],   \
                                            uint tg [[threadgroup_position_in_grid]],  \
                                            ushort tiisg [[thread_index_in_simdgroup]], \
                                            ushort sg [[simdgroup_index_in_threadgroup]]);

SCALED_HEAD_ENTRY("q6k_head_logits_scaled", Q6K, block_q6_K)
SCALED_HEAD_ENTRY("a4_head_logits_scaled", A4, block_mlx_a4)
SCALED_HEAD_ENTRY("a8_head_logits_scaled", A8, block_mlx_a8)
SCALED_HEAD_ENTRY("a4h_head_logits_scaled", A4H, block_mlx_a4)
SCALED_HEAD_ENTRY("a8h_head_logits_scaled", A8H, block_mlx_a8)
SCALED_HEAD_ENTRY("q8_0_head_logits_scaled", Q8_0, block_q8_0)
SCALED_HEAD_ENTRY("q5k_head_logits_scaled", Q5K, block_q5_K)

template [[host_name("q6k_head_logits")]]
kernel void head_logits_kernel<Q6K>(device const block_q6_K *weights [[buffer(0)]],
                                    device const float *x [[buffer(1)]],
                                    device const float *gain [[buffer(2)]],
                                    device float *logits [[buffer(3)]],
                                    device atomic_uint *faults [[buffer(4)]],
                                    constant MatvecArgs &args [[buffer(5)]],
                                    uint tg [[threadgroup_position_in_grid]],
                                    ushort tiisg [[thread_index_in_simdgroup]],
                                    ushort sg [[simdgroup_index_in_threadgroup]]);

template [[host_name("a4_head_logits")]]
kernel void head_logits_kernel<A4>(device const block_mlx_a4 *weights [[buffer(0)]],
                                   device const float *x [[buffer(1)]],
                                   device const float *gain [[buffer(2)]],
                                   device float *logits [[buffer(3)]],
                                   device atomic_uint *faults [[buffer(4)]],
                                   constant MatvecArgs &args [[buffer(5)]],
                                   uint tg [[threadgroup_position_in_grid]],
                                   ushort tiisg [[thread_index_in_simdgroup]],
                                   ushort sg [[simdgroup_index_in_threadgroup]]);

template [[host_name("a8_head_logits")]]
kernel void head_logits_kernel<A8>(device const block_mlx_a8 *weights [[buffer(0)]],
                                   device const float *x [[buffer(1)]],
                                   device const float *gain [[buffer(2)]],
                                   device float *logits [[buffer(3)]],
                                   device atomic_uint *faults [[buffer(4)]],
                                   constant MatvecArgs &args [[buffer(5)]],
                                   uint tg [[threadgroup_position_in_grid]],
                                   ushort tiisg [[thread_index_in_simdgroup]],
                                   ushort sg [[simdgroup_index_in_threadgroup]]);

template [[host_name("a4h_head_logits")]]
kernel void head_logits_kernel<A4H>(device const block_mlx_a4 *weights [[buffer(0)]],
                                    device const float *x [[buffer(1)]],
                                    device const float *gain [[buffer(2)]],
                                    device float *logits [[buffer(3)]],
                                    device atomic_uint *faults [[buffer(4)]],
                                    constant MatvecArgs &args [[buffer(5)]],
                                    uint tg [[threadgroup_position_in_grid]],
                                    ushort tiisg [[thread_index_in_simdgroup]],
                                    ushort sg [[simdgroup_index_in_threadgroup]]);

template [[host_name("a8h_head_logits")]]
kernel void head_logits_kernel<A8H>(device const block_mlx_a8 *weights [[buffer(0)]],
                                    device const float *x [[buffer(1)]],
                                    device const float *gain [[buffer(2)]],
                                    device float *logits [[buffer(3)]],
                                    device atomic_uint *faults [[buffer(4)]],
                                    constant MatvecArgs &args [[buffer(5)]],
                                    uint tg [[threadgroup_position_in_grid]],
                                    ushort tiisg [[thread_index_in_simdgroup]],
                                    ushort sg [[simdgroup_index_in_threadgroup]]);

template [[host_name("q8_0_head_logits")]]
kernel void head_logits_kernel<Q8_0>(device const block_q8_0 *weights [[buffer(0)]],
                                     device const float *x [[buffer(1)]],
                                     device const float *gain [[buffer(2)]],
                                     device float *logits [[buffer(3)]],
                                     device atomic_uint *faults [[buffer(4)]],
                                     constant MatvecArgs &args [[buffer(5)]],
                                     uint tg [[threadgroup_position_in_grid]],
                                     ushort tiisg [[thread_index_in_simdgroup]],
                                     ushort sg [[simdgroup_index_in_threadgroup]]);

template [[host_name("q5k_head_logits")]]
kernel void head_logits_kernel<Q5K>(device const block_q5_K *weights [[buffer(0)]],
                                    device const float *x [[buffer(1)]],
                                    device const float *gain [[buffer(2)]],
                                    device float *logits [[buffer(3)]],
                                    device atomic_uint *faults [[buffer(4)]],
                                    constant MatvecArgs &args [[buffer(5)]],
                                    uint tg [[threadgroup_position_in_grid]],
                                    ushort tiisg [[thread_index_in_simdgroup]],
                                    ushort sg [[simdgroup_index_in_threadgroup]]);

// The second dispatch's arguments: the rows the first one stored, and the
// layer a raise is recorded in.
struct ArgmaxArgs {
    uint rows;
    uint layer;
};

static_assert(sizeof(ArgmaxArgs) == 8, "ArgmaxArgs is 8 bytes: the host packs two words");

// One logit into a thread's running best. A non-finite logit raises the logit
// site; a NaN never wins. The rows of a thread arrive in order, so the first
// row keeps an equal value.
static inline void head_take(float v, uint row, thread float &best_v, thread uint &best_i,
                             device atomic_uint *faults, constant ArgmaxArgs &args)
{
    if (!isfinite(v)) {
        raise(faults, args.layer, BLOOMERY_FAULT_SITE_LOGIT);
    }
    if (v > best_v) {
        best_v = v;
        best_i = row;
    }
}

// One threadgroup over the `rows` logits. A thread reads four logits a load
// (quad q is rows 4q to 4q + 3) and keeps four loads in flight; the rows past
// the last whole quad are taken one a thread after the quads, so a thread's
// rows still reach head_take in order. The launch checks that the logits
// start on 16 bytes.
[[max_total_threads_per_threadgroup(32u * HEAD_STAGE_MAX)]]
kernel void logits_argmax(device const float *logits [[buffer(0)]],
                          device uint *step_id [[buffer(1)]],
                          device atomic_uint *faults [[buffer(2)]],
                          constant ArgmaxArgs &args [[buffer(3)]],
                          uint tid [[thread_position_in_threadgroup]],
                          ushort tiisg [[thread_index_in_simdgroup]],
                          ushort sg [[simdgroup_index_in_threadgroup]])
{
    threadgroup ulong stage[HEAD_STAGE_MAX];

    float best_v = -INFINITY;
    uint best_i = 0u;
    device const float4 *quads = (device const float4 *)logits;
    const uint nq = args.rows / 4u;
    uint q = tid;
    for (; q < nq && nq - q > 3u * HEAD_THREADS; q += 4u * HEAD_THREADS) {
        float4 v[4];
        FOR_UNROLL (short u = 0; u < 4; ++u) {
            v[u] = quads[q + (uint)u * HEAD_THREADS];
        }
        FOR_UNROLL (short u = 0; u < 4; ++u) {
            const uint row = 4u * (q + (uint)u * HEAD_THREADS);
            FOR_UNROLL (short c = 0; c < 4; ++c) {
                head_take(v[u][c], row + (uint)c, best_v, best_i, faults, args);
            }
        }
    }
    for (; q < nq; q += HEAD_THREADS) {
        const float4 v = quads[q];
        FOR_UNROLL (short c = 0; c < 4; ++c) {
            head_take(v[c], 4u * q + (uint)c, best_v, best_i, faults, args);
        }
    }
    const uint row = 4u * nq + tid;
    if (row < args.rows) {
        head_take(logits[row], row, best_v, best_i, faults, args);
    }

    ulong best_key = ticket_max_key(argmax_key(best_v, best_i), stage, (uint)tiisg, (uint)sg,
                                    HEAD_SIMD_GROUPS);
    if (tid == 0u) {
        // (-inf, 0) is the smallest key, so any finite logit wins. When it
        // wins, every logit was NaN and the id must not be a token.
        ulong empty_key = argmax_key(-INFINITY, 0u);
        if (best_key == empty_key) {
            raise(faults, args.layer, BLOOMERY_FAULT_SITE_LOGIT);
            step_id[0] = 0xFFFFFFFFu;
        } else {
            step_id[0] = ~uint(best_key);
        }
    }
}
