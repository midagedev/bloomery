// The A8 entries of the quantized mat-vec: instantiations of the row loop
// matvec_q4k.metal owns over MLX's affine 8-bit decode owner (quant.metal),
// under the names the host launches:
//
//   a8_matvec                y = W . x
//   a8_matvec_norm           y = W . rms_norm(x, gain, eps)
//   a8_matvec_resid          y = y + W . x, in place
//   a8_matvec_seg            the row segments of up to four tensors of one
//                            type and one width, in one dispatch
//   a8_matvec_headnorm_resid y = y + W . (g * head_scale): the gated delta
//                            net's out projection — one 64-value group sits
//                            whole inside one value head, so the owner's
//                            head form applies one scale a block
//
// This file comes after matvec_q4k.metal (its row loop and entry templates)
// and quant.metal.

#include <metal_stdlib>

using namespace metal;

#pragma METAL fp contract(off)

template [[host_name("a8_matvec")]]
kernel void matvec_kernel<A8, false>(device const block_mlx_a8 * W [[buffer(0)]],
                                     device const float      * x [[buffer(1)]],
                                     device float            * y [[buffer(2)]],
                                     constant MatvecArgs     & args [[buffer(3)]],
                                     uint3  tgpig [[threadgroup_position_in_grid]],
                                     ushort tiisg [[thread_index_in_simdgroup]],
                                     ushort sgitg [[simdgroup_index_in_threadgroup]]);

template [[host_name("a8_matvec_resid")]]
kernel void matvec_kernel<A8, true>(device const block_mlx_a8 * W [[buffer(0)]],
                                    device const float      * x [[buffer(1)]],
                                    device float            * y [[buffer(2)]],
                                    constant MatvecArgs     & args [[buffer(3)]],
                                    uint3  tgpig [[threadgroup_position_in_grid]],
                                    ushort tiisg [[thread_index_in_simdgroup]],
                                    ushort sgitg [[simdgroup_index_in_threadgroup]]);

template [[host_name("a8_matvec_norm")]]
kernel void matvec_norm_kernel<A8>(device const block_mlx_a8 * W [[buffer(0)]],
                                   device const float      * x [[buffer(1)]],
                                   device float            * y [[buffer(2)]],
                                   constant MatvecArgs     & args [[buffer(3)]],
                                   device const float      * gain [[buffer(4)]],
                                   device atomic_uint      * faults [[buffer(5)]],
                                   uint3  tgpig [[threadgroup_position_in_grid]],
                                   ushort tiisg [[thread_index_in_simdgroup]],
                                   ushort sgitg [[simdgroup_index_in_threadgroup]]);

SCALED_SEG_ENTRIES("a8", A8, block_mlx_a8)

template [[host_name("a8_matvec_seg")]]
kernel void matvec_seg_kernel<A8>(device const block_mlx_a8 * W [[buffer(0)]],
                                  device const float      * x [[buffer(1)]],
                                  device float            * y [[buffer(2)]],
                                  constant SegArgs        & args [[buffer(3)]],
                                  device const float      * gain [[buffer(4)]],
                                  device atomic_uint      * faults [[buffer(5)]],
                                  uint3  tgpig [[threadgroup_position_in_grid]],
                                  ushort tiisg [[thread_index_in_simdgroup]],
                                  ushort sgitg [[simdgroup_index_in_threadgroup]]);

template [[host_name("a8_matvec_headnorm_resid")]]
kernel void matvec_headnorm_kernel<A8>(device const block_mlx_a8 * W [[buffer(0)]],
                                       device const float      * g [[buffer(1)]],
                                       device const float      * partials [[buffer(2)]],
                                       device float            * y [[buffer(3)]],
                                       constant MatvecArgs     & args [[buffer(4)]],
                                       device atomic_uint      * faults [[buffer(5)]],
                                       uint3  tgpig [[threadgroup_position_in_grid]],
                                       ushort tiisg [[thread_index_in_simdgroup]],
                                       ushort sgitg [[simdgroup_index_in_threadgroup]]);
