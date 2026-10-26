// The output head: one quantized mat-vec with the norm on the load, and the
// argmax in the same dispatch. The body is a template over the quant
// (quant.metal), its row loop matvec_q4k.metal's; the entries are the Q6_K,
// A4, Q8_0 and Q5_K instantiations, under their host names. A threadgroup
// owns the rows of its BLOOMERY_MATVEC_SIMD_GROUPS SIMD groups,
// BLOOMERY_MATVEC_ROWS_PER_SIMD_GROUP each. It writes one packed candidate.
// The last threadgroup's threads together reduce every candidate and thread
// 0 stores the winning row index.
//
// The packed key is ticket.metal's argmax_key. An empty group, and a group
// whose live logits are all NaN, packs (-inf, 0). When that key wins, the
// step id is not a token.
//
// This file comes after matvec_q6k.metal (the entry templates) and
// ticket.metal.

#include <metal_stdlib>

using namespace metal;

#pragma METAL fp contract(off)

// The head's threadgroup: the mat-vec function constant, the count the head's
// pipeline is specialized at, so the strides read the same count the launch
// dispatches. A function constant cannot size an array, so the stage is sized
// at the most a threadgroup holds (geometry.rs's
// MAX_SIMD_GROUPS_PER_GROUP: the device's 1024 threads, 32 of them in a
// group) and reduced over the count the launch runs.
#define HEAD_SIMD_GROUPS BLOOMERY_MATVEC_SIMD_GROUPS
#define HEAD_THREADS (32u * HEAD_SIMD_GROUPS)
#define HEAD_STAGE_MAX 32u

template <typename Q>
kernel void head_argmax_kernel(device const typename Q::Block *weights [[buffer(0)]],
                               device const float *x [[buffer(1)]],
                               device const float *gain [[buffer(2)]],
                               device float *logits [[buffer(3)]],
                               device coherent(device) ulong *candidates [[buffer(4)]],
                               device atomic_uint *ticket [[buffer(5)]],
                               device uint *step_id [[buffer(6)]],
                               device atomic_uint *faults [[buffer(7)]],
                               constant MatvecArgs &args [[buffer(8)]],
                               uint tg [[threadgroup_position_in_grid]],
                               uint tid [[thread_position_in_threadgroup]],
                               ushort tiisg [[thread_index_in_simdgroup]],
                               ushort sg [[simdgroup_index_in_threadgroup]],
                               uint groups [[threadgroups_per_grid]])
{
    threadgroup uint last_flag = 0u;
    threadgroup ulong stage[HEAD_STAGE_MAX];

    const int first_row = simd_first_row(tg, sg);
    matvec_rows<Q, true, false, false, false>(weights, x, logits, (int)args.nblocks, args.eps,
                                              args.layer, gain, nullptr, faults, first_row,
                                              (int)args.rows, tiisg);

    // Every logit this group stored is visible.
    threadgroup_barrier(mem_flags::mem_device);

    // Lane 0 of a SIMD group raises on a non-finite logit and leaves the NaN
    // stored. The following barrier publishes those stores to the thread
    // that reduces.
    if (tiisg == 0u) {
        for (int r = 0; r < (int)BLOOMERY_MATVEC_ROWS_PER_SIMD_GROUP; ++r) {
            int row = first_row + r;
            if (row >= (int)args.rows) {
                break;
            }
            if (!isfinite(logits[row])) {
                raise(faults, args.layer, BLOOMERY_FAULT_SITE_LOGIT);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_device);

    if (tid == 0u) {
        const int rows_per_group = (int)BLOOMERY_MATVEC_SIMD_GROUPS *
                                   (int)BLOOMERY_MATVEC_ROWS_PER_SIMD_GROUP;
        const int base = (int)tg * rows_per_group;
        float best_v = -INFINITY;
        uint best_i = 0u;
        for (int r = 0; r < rows_per_group; ++r) {
            int row = base + r;
            if (row >= (int)args.rows) {
                break;
            }
            float v = logits[row];
            if (v > best_v || (v == best_v && (uint)row < best_i)) {
                best_v = v;
                best_i = (uint)row;
            }
        }
        ticket_write_u64(candidates, tg, argmax_key(best_v, best_i));
    }

    ticket_is_last(ticket, groups, tid, last_flag);
    if (last_flag == 1u) {
        // The threads of the last threadgroup share the candidates by
        // stride, so the tail is one strided load each, not one thread's
        // chain. A share with no candidate holds key 0, which no packed key
        // is, so it never wins.
        ticket_acquire();
        ulong mine = 0u;
        for (uint g = tid; g < groups; g += HEAD_THREADS) {
            ulong key = ticket_load_u64(candidates, g);
            if (key > mine) {
                mine = key;
            }
        }
        ulong best_key = ticket_max_key(mine, stage, (uint)tiisg, (uint)sg,
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
            ticket_reset(ticket);
        }
    }
}

template [[host_name("q6k_head_argmax")]]
kernel void head_argmax_kernel<Q6K>(device const block_q6_K *weights [[buffer(0)]],
                                    device const float *x [[buffer(1)]],
                                    device const float *gain [[buffer(2)]],
                                    device float *logits [[buffer(3)]],
                                    device coherent(device) ulong *candidates [[buffer(4)]],
                                    device atomic_uint *ticket [[buffer(5)]],
                                    device uint *step_id [[buffer(6)]],
                                    device atomic_uint *faults [[buffer(7)]],
                                    constant MatvecArgs &args [[buffer(8)]],
                                    uint tg [[threadgroup_position_in_grid]],
                                    uint tid [[thread_position_in_threadgroup]],
                                    ushort tiisg [[thread_index_in_simdgroup]],
                                    ushort sg [[simdgroup_index_in_threadgroup]],
                                    uint groups [[threadgroups_per_grid]]);

template [[host_name("a4_head_argmax")]]
kernel void head_argmax_kernel<A4>(device const block_mlx_a4 *weights [[buffer(0)]],
                                   device const float *x [[buffer(1)]],
                                   device const float *gain [[buffer(2)]],
                                   device float *logits [[buffer(3)]],
                                   device coherent(device) ulong *candidates [[buffer(4)]],
                                   device atomic_uint *ticket [[buffer(5)]],
                                   device uint *step_id [[buffer(6)]],
                                   device atomic_uint *faults [[buffer(7)]],
                                   constant MatvecArgs &args [[buffer(8)]],
                                   uint tg [[threadgroup_position_in_grid]],
                                   uint tid [[thread_position_in_threadgroup]],
                                   ushort tiisg [[thread_index_in_simdgroup]],
                                   ushort sg [[simdgroup_index_in_threadgroup]],
                                   uint groups [[threadgroups_per_grid]]);

template [[host_name("q8_0_head_argmax")]]
kernel void head_argmax_kernel<Q8_0>(device const block_q8_0 *weights [[buffer(0)]],
                                     device const float *x [[buffer(1)]],
                                     device const float *gain [[buffer(2)]],
                                     device float *logits [[buffer(3)]],
                                     device coherent(device) ulong *candidates [[buffer(4)]],
                                     device atomic_uint *ticket [[buffer(5)]],
                                     device uint *step_id [[buffer(6)]],
                                     device atomic_uint *faults [[buffer(7)]],
                                     constant MatvecArgs &args [[buffer(8)]],
                                     uint tg [[threadgroup_position_in_grid]],
                                     uint tid [[thread_position_in_threadgroup]],
                                     ushort tiisg [[thread_index_in_simdgroup]],
                                     ushort sg [[simdgroup_index_in_threadgroup]],
                                     uint groups [[threadgroups_per_grid]]);

template [[host_name("q5k_head_argmax")]]
kernel void head_argmax_kernel<Q5K>(device const block_q5_K *weights [[buffer(0)]],
                                    device const float *x [[buffer(1)]],
                                    device const float *gain [[buffer(2)]],
                                    device float *logits [[buffer(3)]],
                                    device coherent(device) ulong *candidates [[buffer(4)]],
                                    device atomic_uint *ticket [[buffer(5)]],
                                    device uint *step_id [[buffer(6)]],
                                    device atomic_uint *faults [[buffer(7)]],
                                    constant MatvecArgs &args [[buffer(8)]],
                                    uint tg [[threadgroup_position_in_grid]],
                                    uint tid [[thread_position_in_threadgroup]],
                                    ushort tiisg [[thread_index_in_simdgroup]],
                                    ushort sg [[simdgroup_index_in_threadgroup]],
                                    uint groups [[threadgroups_per_grid]]);
