// The ticket: the one owner of the cross-threadgroup handoff. The router
// calls the float pair. The head calls the u64 pair for a packed candidate,
// and the same arrive, acquire and reset.
//
// The contract is these functions, in this order:
//   ticket_write      one producing lane stores one partial
//   ticket_write_u64  one producing lane stores one packed candidate
//   (the caller then barriers the threadgroup on device memory, so every
//    partial that threadgroup wrote is visible to the thread that arrives)
//   ticket_arrive     that one thread fences, then a relaxed fetch_add; the
//                     value returned is the old counter, and the threadgroup
//                     is last when old + 1 equals the dispatch's group count
//   ticket_acquire    the last threadgroup's reader fences before it loads
//   ticket_load       that reader loads one partial through coherent(device)
//   ticket_load_u64   that reader loads one packed candidate the same way
//   ticket_reset      that reader stores 0, relaxed
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

// A packed candidate. The same contract as the float pair: a plain
// coherent(device) store and load, no atomic.
static void ticket_write_u64(device coherent(device) ulong *candidates, uint index, ulong value)
{
    candidates[index] = value;
}

static ulong ticket_load_u64(device coherent(device) ulong *candidates, uint index)
{
    return candidates[index];
}
