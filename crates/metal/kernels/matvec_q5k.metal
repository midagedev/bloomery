// The Q5_K entries of the quantized mat-vec: instantiations of the row loop
// matvec_q4k.metal owns over GGUF's Q5_K decode owner (quant.metal), under
// the names the host launches:
//
//   q5k_matvec                y = W . x
//   q5k_matvec_norm           y = W . rms_norm(x, gain, eps)
//   q5k_matvec_resid          y = y + W . x, in place
//   q5k_matvec_seg            the row segments of up to four tensors of one
//                             type and one width, in one dispatch
//   q5k_matvec_headnorm_resid y = y + W . (g * head_scale): the gated delta
//                             net's out projection — a 256-value block spans
//                             two value heads, so the owner's head form
//                             scales each half once, as Q4_K's
//
// This file comes after matvec_q4k.metal (its row loop and entry templates)
// and quant.metal.

#include <metal_stdlib>

using namespace metal;

#pragma METAL fp contract(off)

template [[host_name("q5k_matvec")]]
kernel void matvec_kernel<Q5K, false>(device const block_q5_K * W [[buffer(0)]],
                                      device const float      * x [[buffer(1)]],
                                      device float            * y [[buffer(2)]],
                                      constant MatvecArgs     & args [[buffer(3)]],
                                      uint3  tgpig [[threadgroup_position_in_grid]],
                                      ushort tiisg [[thread_index_in_simdgroup]],
                                      ushort sgitg [[simdgroup_index_in_threadgroup]]);

template [[host_name("q5k_matvec_resid")]]
kernel void matvec_kernel<Q5K, true>(device const block_q5_K * W [[buffer(0)]],
                                     device const float      * x [[buffer(1)]],
                                     device float            * y [[buffer(2)]],
                                     constant MatvecArgs     & args [[buffer(3)]],
                                     uint3  tgpig [[threadgroup_position_in_grid]],
                                     ushort tiisg [[thread_index_in_simdgroup]],
                                     ushort sgitg [[simdgroup_index_in_threadgroup]]);

template [[host_name("q5k_matvec_norm")]]
kernel void matvec_norm_kernel<Q5K>(device const block_q5_K * W [[buffer(0)]],
                                    device const float      * x [[buffer(1)]],
                                    device float            * y [[buffer(2)]],
                                    constant MatvecArgs     & args [[buffer(3)]],
                                    device const float      * gain [[buffer(4)]],
                                    device atomic_uint      * faults [[buffer(5)]],
                                    uint3  tgpig [[threadgroup_position_in_grid]],
                                    ushort tiisg [[thread_index_in_simdgroup]],
                                    ushort sgitg [[simdgroup_index_in_threadgroup]]);

SCALED_SEG_ENTRIES("q5k", Q5K, block_q5_K)

template [[host_name("q5k_matvec_seg")]]
kernel void matvec_seg_kernel<Q5K>(device const block_q5_K * W [[buffer(0)]],
                                   device const float      * x [[buffer(1)]],
                                   device float            * y [[buffer(2)]],
                                   constant SegArgs        & args [[buffer(3)]],
                                   device const float      * gain [[buffer(4)]],
                                   device atomic_uint      * faults [[buffer(5)]],
                                   uint3  tgpig [[threadgroup_position_in_grid]],
                                   ushort tiisg [[thread_index_in_simdgroup]],
                                   ushort sgitg [[simdgroup_index_in_threadgroup]]);

template [[host_name("q5k_matvec_headnorm_resid")]]
kernel void matvec_headnorm_kernel<Q5K>(device const block_q5_K * W [[buffer(0)]],
                                        device const float      * g [[buffer(1)]],
                                        device const float      * partials [[buffer(2)]],
                                        device float            * y [[buffer(3)]],
                                        constant MatvecArgs     & args [[buffer(4)]],
                                        device atomic_uint      * faults [[buffer(5)]],
                                        uint3  tgpig [[threadgroup_position_in_grid]],
                                        ushort tiisg [[thread_index_in_simdgroup]],
                                        ushort sgitg [[simdgroup_index_in_threadgroup]]);
