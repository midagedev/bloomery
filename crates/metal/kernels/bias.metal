// An attention layer's query, key and value biases: `y[r][i] += bias[i]` over
// `rows` rows of `cols4` float4s, `y` a P1 output whose rows are `stride4`
// float4s apart. The add is one f32 add a value, so a row's bits are the
// projection's output plus the bias, rounded once.

#include <metal_stdlib>

using namespace metal;

#pragma METAL fp contract(off)

struct BiasRowsArgs {
    uint cols4;
    uint stride4;
};

kernel void bias_rows(device float4 *y [[buffer(0)]],
                      const device float4 *bias [[buffer(1)]],
                      constant BiasRowsArgs &args [[buffer(2)]],
                      uint2 gid [[thread_position_in_grid]])
{
    if (gid.x < args.cols4) {
        y[gid.y * args.stride4 + gid.x] += bias[gid.x];
    }
}
