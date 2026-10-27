// The f32 entry of the quantized mat-vec: the segmented dispatch over quant.metal's
// F32 owner, under the name the host launches:
//
//   f32_matvec_seg   the row segments of up to four tensors of one type and
//                    one width, in one dispatch: a layer's f32 projections
//                    (the gated delta net's alpha and beta)
//
// The plain, norm, residual and head-norm modes have no f32 entry: no site
// runs them. This file comes after matvec_q4k.metal (its row loop and entry
// templates) and quant.metal.

#include <metal_stdlib>

using namespace metal;

#pragma METAL fp contract(off)

template [[host_name("f32_matvec_seg")]]
kernel void matvec_seg_kernel<F32>(device const block_f32_16 * W [[buffer(0)]],
                                   device const float      * x [[buffer(1)]],
                                   device float            * y [[buffer(2)]],
                                   constant SegArgs        & args [[buffer(3)]],
                                   device const float      * gain [[buffer(4)]],
                                   device atomic_uint      * faults [[buffer(5)]],
                                   uint3  tgpig [[threadgroup_position_in_grid]],
                                   ushort tiisg [[thread_index_in_simdgroup]],
                                   ushort sgitg [[simdgroup_index_in_threadgroup]]);
