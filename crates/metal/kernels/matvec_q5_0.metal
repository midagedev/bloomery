// The Q5_0 and Q5_1 entries of the quantized mat-vec: instantiations of the row
// loop matvec_q4k.metal owns over the decode owners of quant_b32.metal, under
// the names the host launches:
//
//   q5_0_matvec        y = W . x
//   q5_0_matvec_norm   y = W . rms_norm(x, gain, eps)
//   q5_0_matvec_resid  y = y + W . x, in place
//   q5_0_matvec_seg    the row segments of up to four tensors of one type
//                      and one width, in one dispatch
//   q5_1_matvec / _norm / _resid / _seg   the same over Q5_1
//
// Neither type has a head-normed entry: no gated delta net projection is
// stored in them. The embeddings are in embed.metal.
//
// This file comes after matvec_q4k.metal (its row loop and entry templates)
// and quant_b32.metal.

#include <metal_stdlib>

using namespace metal;

#pragma METAL fp contract(off)

#define B32_MATVEC_ENTRIES(Q, BLOCK, STORE, NORM, RESID, SEG)                         \
    template [[host_name(STORE)]]                                                     \
    kernel void matvec_kernel<Q, false>(device const BLOCK * W [[buffer(0)]],         \
                                        device const float * x [[buffer(1)]],         \
                                        device float       * y [[buffer(2)]],         \
                                        constant MatvecArgs & args [[buffer(3)]],     \
                                        uint3  tgpig [[threadgroup_position_in_grid]], \
                                        ushort tiisg [[thread_index_in_simdgroup]],   \
                                        ushort sgitg [[simdgroup_index_in_threadgroup]]); \
    template [[host_name(RESID)]]                                                     \
    kernel void matvec_kernel<Q, true>(device const BLOCK * W [[buffer(0)]],          \
                                       device const float * x [[buffer(1)]],          \
                                       device float       * y [[buffer(2)]],          \
                                       constant MatvecArgs & args [[buffer(3)]],      \
                                       uint3  tgpig [[threadgroup_position_in_grid]], \
                                       ushort tiisg [[thread_index_in_simdgroup]],    \
                                       ushort sgitg [[simdgroup_index_in_threadgroup]]); \
    template [[host_name(NORM)]]                                                      \
    kernel void matvec_norm_kernel<Q>(device const BLOCK * W [[buffer(0)]],           \
                                      device const float * x [[buffer(1)]],           \
                                      device float       * y [[buffer(2)]],           \
                                      constant MatvecArgs & args [[buffer(3)]],       \
                                      device const float * gain [[buffer(4)]],        \
                                      device atomic_uint * faults [[buffer(5)]],      \
                                      uint3  tgpig [[threadgroup_position_in_grid]],  \
                                      ushort tiisg [[thread_index_in_simdgroup]],     \
                                      ushort sgitg [[simdgroup_index_in_threadgroup]]); \
    template [[host_name(SEG)]]                                                       \
    kernel void matvec_seg_kernel<Q>(device const BLOCK * W [[buffer(0)]],            \
                                     device const float * x [[buffer(1)]],            \
                                     device float       * y [[buffer(2)]],            \
                                     constant SegArgs   & args [[buffer(3)]],         \
                                     device const float * gain [[buffer(4)]],         \
                                     device atomic_uint * faults [[buffer(5)]],       \
                                     uint3  tgpig [[threadgroup_position_in_grid]],   \
                                     ushort tiisg [[thread_index_in_simdgroup]],      \
                                     ushort sgitg [[simdgroup_index_in_threadgroup]]);

B32_MATVEC_ENTRIES(Q5_0, block_q5_0, "q5_0_matvec", "q5_0_matvec_norm", "q5_0_matvec_resid", "q5_0_matvec_seg")
B32_MATVEC_ENTRIES(Q5_1, block_q5_1, "q5_1_matvec", "q5_1_matvec_norm", "q5_1_matvec_resid", "q5_1_matvec_seg")
