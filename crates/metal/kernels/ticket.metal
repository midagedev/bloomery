// The ticket: the one owner of the cross-threadgroup handoff. The router and
// the attention fold call the float pair, the arrive, the acquire and the
// reset. The argmax key and the threadgroup's reduction of it live here too:
// the router and the output head's argmax both pack and reduce keys.
//
// The contract is these functions, in this order:
//   ticket_write      one producing lane stores one partial
//   (the caller then barriers the threadgroup on device memory, so every
//    partial that threadgroup wrote is visible to the thread that arrives)
//   ticket_arrive     that one thread fences, then a relaxed fetch_add; the
//                     value returned is the old counter, and the threadgroup
//                     is last when old + 1 equals the dispatch's group count
//   ticket_acquire    the last threadgroup's readers fence before they load
//   ticket_load       a reader loads its partials through coherent(device)
//   ticket_reset      that reader stores 0, relaxed
//   ticket_is_last    the producing lanes have stored; one barrier publishes
//                     them, thread 0 arrives, and a second barrier publishes
//                     last_flag. The caller does not barrier around it.
//
// The fence is device-scope and sequentially consistent. The fetch_add and
// the reset are relaxed. The partial pointer is coherent(device).

#include <metal_stdlib>

using namespace metal;

#pragma METAL fp contract(off)

static void ticket_write(device coherent(device) float *partials, uint index, float value)
{
    partials[index] = value;
}

static uint ticket_arrive(device atomic_uint *ticket)
{
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    return atomic_fetch_add_explicit(ticket, 1u, memory_order_relaxed);
}

static void ticket_acquire()
{
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
}

static float ticket_load(device coherent(device) float *partials, uint index)
{
    return partials[index];
}

static void ticket_reset(device atomic_uint *ticket)
{
    atomic_store_explicit(ticket, 0u, memory_order_relaxed);
}

// The last threadgroup of the dispatch. `groups` is the grid's threadgroup
// count. Thread 0 writes `last_flag`; every thread reads it after the return.
static void ticket_is_last(device atomic_uint *ticket, uint groups, uint tid,
                           threadgroup uint &last_flag)
{
    threadgroup_barrier(mem_flags::mem_device);
    if (tid == 0u) {
        uint old = ticket_arrive(ticket);
        last_flag = (old + 1u == groups) ? 1u : 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// The argmax key: the value's order bits over the id's complement. v == 0
// stores bits 0, so -0 and +0 are one key. A negative bit pattern is bitwise
// complemented. A non-negative one is OR-ed with 0x80000000. The key is that
// u32 in the high half and ~index in the low half, so an unsigned maximum of
// keys is the comparison "the larger value first, at an equal value the
// smaller id", and a reduction of keys needs no order: the maximum is
// associative and commutative.
static ulong argmax_key(float v, uint i)
{
    uint b = v == 0.f ? 0u : as_type<uint>(v);
    uint o = (b & 0x80000000u) != 0u ? ~b : (b | 0x80000000u);
    return ((ulong)o << 32) | (ulong)(~i);
}

// The threadgroup's largest packed key. Every thread arrives holding the
// largest key of its own share and the same value leaves. The SIMD group
// reduces in two halves, because the SIMD-group maximum takes no 64-bit
// integer: the high half's maximum, then the low halves' maximum of the lanes
// that hold it, is the keys' maximum, the key ordering the high half before
// the low. A lane that does not hold the high half's maximum contributes a
// low of 0, which no packed id is. One slot per SIMD group then holds the
// group's key, and every thread reads them all. The caller barriers on
// threadgroup memory before it reuses `stage` after the return.
static ulong ticket_max_key(ulong key, threadgroup ulong *stage,
                            uint lane, uint sg, uint sg_count)
{
    uint o = uint(key >> 32);
    uint hi = simd_max(o);
    uint lo = simd_max(o == hi ? uint(key) : 0u);
    if (lane == 0u) {
        stage[sg] = ((ulong)hi << 32) | (ulong)lo;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ulong best = stage[0];
    for (uint g = 1u; g < sg_count; ++g) {
        best = max(best, stage[g]);
    }
    return best;
}
