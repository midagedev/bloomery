// The GEMM staging's decode, run on its own: each thread stages one 16-value
// chunk of a block through the owner's `dequant16`, as the prompt GEMMs do, and
// writes the 16 values as f32. The gate appends this to the kernels' source and
// holds the values to the reader's.

template <typename Q>
kernel void b32_dequant16_probe(device const typename Q::Block * blocks [[buffer(0)]],
                                device float                    * out    [[buffer(1)]],
                                constant uint                   & chunks [[buffer(2)]],
                                uint gid [[thread_position_in_grid]]) {
    if (gid >= chunks) {
        return;
    }
    matrix<float, 4, 4> reg;
    Q::template dequant16<float>(blocks[gid / 2], (short)(gid % 2), reg);
    for (short j = 0; j < 16; ++j) {
        out[gid * 16 + j] = reg[j / 4][j % 4];
    }
}

template [[host_name("b32_dequant16_q5_0")]]
kernel void b32_dequant16_probe<Q5_0>(device const block_q5_0 * blocks [[buffer(0)]],
                                      device float            * out    [[buffer(1)]],
                                      constant uint           & chunks [[buffer(2)]],
                                      uint gid [[thread_position_in_grid]]);

template [[host_name("b32_dequant16_q5_1")]]
kernel void b32_dequant16_probe<Q5_1>(device const block_q5_1 * blocks [[buffer(0)]],
                                      device float            * out    [[buffer(1)]],
                                      constant uint           & chunks [[buffer(2)]],
                                      uint gid [[thread_position_in_grid]]);
