// The output head: one Q6_K mat-vec with the norm on the load, and the
// argmax in the same dispatch. A threadgroup owns sixteen rows (eight SIMD
// groups, two rows each). It writes one packed candidate. The last
// threadgroup reads every candidate and stores the winning row index.
//
// The packed key is CUDA's argmax_key. v == 0 stores bits 0, so -0 and +0
// are one key. A negative bit pattern is bitwise complemented. A
// non-negative one is OR-ed with 0x80000000. The key is that u32 in the
// high half and ~index in the low half. An unsigned maximum of keys is the
// comparison below, so the scan order does not matter. An empty group, and
// a group whose live logits are all NaN, packs (-inf, 0).
//
// This file comes after matvec_q6k.metal (q6k_rows, block_q6_K) and
// ticket.metal.

#include <metal_stdlib>

using namespace metal;

#pragma METAL fp contract(off)

static ulong argmax_key(float v, uint i)
{
    uint b = v == 0.f ? 0u : as_type<uint>(v);
    uint o = (b & 0x80000000u) != 0u ? ~b : (b | 0x80000000u);
    return ((ulong)o << 32) | (ulong)(~i);
}

kernel void q6k_head_argmax(device const block_q6_K *weights [[buffer(0)]],
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

    // A launch the host accepted passes this. The bounds mutant flips the
    // comparison so every thread returns before a barrier.
    uint head_bounds_gate = 1u;
    if (head_bounds_gate == 0u) {
        return;
    }

    const int first_row = (int)((tg * BLOOMERY_MATVEC_SIMD_GROUPS + sg) *
                                BLOOMERY_MATVEC_ROWS_PER_SIMD_GROUP);
    q6k_rows<true, false>(weights, x, logits, (int)args.nblocks, args.eps, args.layer, gain,
                          faults, first_row, (int)args.rows, tiisg);

    // Every logit this group stored is visible.
    threadgroup_barrier(mem_flags::mem_device);

    // Lane 0 of a SIMD group is thread 0 of that group's rows. It raises on
    // a non-finite logit and leaves the NaN stored, then scales the stored
    // value. The scale is 1. The following barrier publishes those stores
    // to the thread that reduces.
    if (tiisg == 0u) {
        float head_logit_scale = 1.f;
        for (int r = 0; r < (int)BLOOMERY_MATVEC_ROWS_PER_SIMD_GROUP; ++r) {
            int row = first_row + r;
            if (row >= (int)args.rows) {
                break;
            }
            if (!isfinite(logits[row])) {
                raise(faults, args.layer, BLOOMERY_FAULT_SITE_LOGIT);
            }
            logits[row] *= head_logit_scale;
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
            bool head_takes = v > best_v || (v == best_v && (uint)row < best_i);
            if (head_takes) {
                best_v = v;
                best_i = (uint)row;
            }
        }
        ticket_write_u64(candidates, tg, argmax_key(best_v, best_i));
    }

    threadgroup_barrier(mem_flags::mem_device);
    if (tid == 0u) {
        uint old = ticket_arrive(ticket);
        last_flag = (old + 1u == groups) ? 1u : 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (last_flag == 1u && tid == 0u) {
        ticket_acquire();
        ulong best_key = ticket_load_u64(candidates, 0u);
        for (uint g = 1u; g < groups; ++g) {
            ulong key = ticket_load_u64(candidates, g);
            if (key > best_key) {
                best_key = key;
            }
        }
        uint id = ~uint(best_key);
        step_id[0] = id;
        ticket_reset(ticket);
    }
}
