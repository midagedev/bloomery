// The gate's own kernels: a spin that a second kernel can end, and the
// arithmetic the two compile paths must agree on.

#include <metal_stdlib>

using namespace metal;

#pragma METAL fp contract(off)

// Reads `flag` up to `limit` times, then writes the result: 2 when it saw the
// flag set, 1 when it ran out. The result is written last, so a reader that
// sees it knows the spin is over.
kernel void spin_until_set(device atomic_uint *flag [[buffer(0)]],
                           device uint *out [[buffer(1)]],
                           constant uint &limit [[buffer(2)]])
{
    uint seen = 0;
    for (uint i = 0; i < limit; ++i) {
        if (atomic_load_explicit(flag, memory_order_relaxed) != 0u) {
            seen = 1;
            break;
        }
    }
    out[0] = 1u + seen;
}

kernel void set_flag(device atomic_uint *flag [[buffer(0)]])
{
    atomic_store_explicit(flag, 1u, memory_order_relaxed);
}

// A fused multiply-add, a multiply and an add rounded apart, and two library
// functions: the host checks the first two bit for bit, and the two compile
// paths must agree on all three.
kernel void arith_f32(device float *fused [[buffer(0)]],
                      device float *split [[buffer(1)]],
                      device float *library [[buffer(2)]],
                      const device float *a [[buffer(3)]],
                      const device float *b [[buffer(4)]],
                      const device float *c [[buffer(5)]],
                      constant uint &count [[buffer(6)]],
                      uint gid [[thread_position_in_grid]])
{
    if (gid < count) {
        fused[gid] = fma(a[gid], b[gid], c[gid]);
        split[gid] = a[gid] * b[gid] + c[gid];
        library[gid] = exp(a[gid]) + rsqrt(abs(b[gid]) + 1.0f);
    }
}
