// A typed copy: the round-trip clause's kernel, and the way a shared
// buffer's contents reach a kernel that only knows the device's copy.

#include <metal_stdlib>

using namespace metal;

#pragma METAL fp contract(off)

kernel void copy_f32(device float *dst [[buffer(0)]],
                     const device float *src [[buffer(1)]],
                     constant uint &count [[buffer(2)]],
                     uint gid [[thread_position_in_grid]])
{
    uint i = gid;
    if (i < count) {
        dst[i] = src[i];
    }
}
