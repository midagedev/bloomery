// The Q6_K entries of the quantized mat-vec: instantiations of the row loop
// matvec_q4k.metal owns, under the names the host has always launched, so no
// launcher changes:
//
//   q6k_matvec        y = W . x
//   q6k_matvec_norm   y = W . rms_norm(x, gain, eps)
//   q6k_matvec_resid  y = y + W . x, in place
//   q6k_matvec_seg    the row segments of up to four tensors of one type and
//                     one width, in one dispatch
//
// Q6_K has no head-norm entry; a mode a quant does not have stays absent.
//
// This file comes after matvec_q4k.metal (its row loop and entry templates)
// and quant.metal.

#include <metal_stdlib>

using namespace metal;

#pragma METAL fp contract(off)

template [[host_name("q6k_matvec")]]
kernel void matvec_kernel<Q6K, false>(device const block_q6_K * W [[buffer(0)]],
                                      device const float      * x [[buffer(1)]],
                                      device float            * y [[buffer(2)]],
                                      constant MatvecArgs     & args [[buffer(3)]],
                                      uint3  tgpig [[threadgroup_position_in_grid]],
                                      ushort tiisg [[thread_index_in_simdgroup]],
                                      ushort sgitg [[simdgroup_index_in_threadgroup]]);

template [[host_name("q6k_matvec_resid")]]
kernel void matvec_kernel<Q6K, true>(device const block_q6_K * W [[buffer(0)]],
                                      device const float      * x [[buffer(1)]],
                                      device float            * y [[buffer(2)]],
                                      constant MatvecArgs     & args [[buffer(3)]],
                                      uint3  tgpig [[threadgroup_position_in_grid]],
                                      ushort tiisg [[thread_index_in_simdgroup]],
                                      ushort sgitg [[simdgroup_index_in_threadgroup]]);

template [[host_name("q6k_matvec_norm")]]
kernel void matvec_norm_kernel<Q6K>(device const block_q6_K * W [[buffer(0)]],
                                    device const float      * x [[buffer(1)]],
                                    device float            * y [[buffer(2)]],
                                    constant MatvecArgs     & args [[buffer(3)]],
                                    device const float      * gain [[buffer(4)]],
                                    device atomic_uint      * faults [[buffer(5)]],
                                    uint3  tgpig [[threadgroup_position_in_grid]],
                                    ushort tiisg [[thread_index_in_simdgroup]],
                                    ushort sgitg [[simdgroup_index_in_threadgroup]]);

SCALED_SEG_ENTRIES("q6k", Q6K, block_q6_K)

template [[host_name("q6k_matvec_seg")]]
kernel void matvec_seg_kernel<Q6K>(device const block_q6_K * W [[buffer(0)]],
                                    device const float      * x [[buffer(1)]],
                                    device float            * y [[buffer(2)]],
                                    constant SegArgs        & args [[buffer(3)]],
                                    device const float      * gain [[buffer(4)]],
                                    device atomic_uint      * faults [[buffer(5)]],
                                    uint3  tgpig [[threadgroup_position_in_grid]],
                                    ushort tiisg [[thread_index_in_simdgroup]],
                                    ushort sgitg [[simdgroup_index_in_threadgroup]]);
