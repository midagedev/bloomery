// The Q6_K mat-vec: the same three entries as matvec_q4k.metal over
// block_q6_K, one dispatch each, and the segmented entry beside them.
//
//   q6k_matvec        y = W . x
//   q6k_matvec_norm   y = W . rms_norm(x, gain, eps)
//   q6k_matvec_resid  y = y + W . x, in place
//   q6k_matvec_seg    the row segments of up to four tensors of one type and
//                     one width, in one dispatch
//
// Adapted from llama.cpp's kernel_mul_mv_q6_K_f32 (MIT licence; see
// THIRD_PARTY_NOTICES.md):
//   ggml/src/ggml-metal/kernels/mul_mv.metal
// The lane map (tid/ix/ip/il), the two-block stride, the unpacking of the
// 6-bit values, the row loop over a SIMD group's rows and the simd_sum
// reduction are llama.cpp's: 2 rows per SIMD group. The SIMD groups in a
// threadgroup are the function constant of norm.metal: 2 for the dense
// mat-vec, 8 for the output head. The tail and the variants are matvec_q4k.metal's:
// a SIMD group past the last row returns, and a row past the last re-reads the
// last row and is never stored.
//
// This file comes after matvec_q4k.metal (QK_K, FOR_UNROLL) and norm.metal.

#include <metal_stdlib>

using namespace metal;

#pragma METAL fp contract(off)

struct block_q6_K {
    uint8_t ql[128];
    uint8_t qh[64];
    char    scales[16];
    half    d;
};

static_assert(sizeof(block_q6_K) == 210, "block_q6_K is 210 bytes");

template <bool NORM, bool RESID>
inline void q6k_rows(device const block_q6_K * W,
                      device const float      * x,
                      device float            * y,
                      int nblocks,
                      float eps,
                      uint layer,
                      device const float      * gain,
                      device atomic_uint      * faults,
                      int first_row,
                      int rows,
                      ushort tiisg) {
    constexpr uint8_t kmask1 = 0x03;
    constexpr uint8_t kmask2 = 0x0C;
    constexpr uint8_t kmask3 = 0x30;
    constexpr uint8_t kmask4 = 0xC0;
    constexpr short NR0 = BLOOMERY_MATVEC_ROWS_PER_SIMD_GROUP;

    const int nb = nblocks;
    if (first_row >= rows) {
        return;
    }
    // Each row's blocks, the last row's for a row past the last.
    device const block_q6_K * rp[NR0];
    for (short r = 0; r < NR0; ++r) {
        rp[r] = W + (ulong)min(first_row + r, rows - 1) * (ulong)nb;
    }

    float sumf[NR0];
    for (short r = 0; r < NR0; ++r) sumf[r] = 0.f;
    float sum_sq = 0.f;

    float yl[16];

    const short tid = tiisg/2;
    const short ix  = tiisg%2;
    const short ip  = tid/8;   // 0 or 1
    const short il  = tid%8;
    const short l0  = 4*il;
    const short is  = 8*ip + l0/16;

    const short y_offset   = 128*ip + l0;
    const short q_offset_l =  64*ip + l0;
    const short q_offset_h =  32*ip + l0;

    for (int i = ix; i < nb; i += 2) {
        device const float * yb = x + i * QK_K + y_offset;
        // The gain is bound by the norm entry only: the others form no pointer
        // from it.
        device const float * gb = NORM ? gain + i * QK_K + y_offset : gain;

        for (short l = 0; l < 4; ++l) {
            float a0 = yb[l +  0];
            float a1 = yb[l + 32];
            float a2 = yb[l + 64];
            float a3 = yb[l + 96];
            if (NORM) {
                sum_sq += (a0 * a0 + a1 * a1) + (a2 * a2 + a3 * a3);
                a0 *= gb[l +  0];
                a1 *= gb[l + 32];
                a2 *= gb[l + 64];
                a3 *= gb[l + 96];
            }
            yl[4*l + 0] = a0;
            yl[4*l + 1] = a1;
            yl[4*l + 2] = a2;
            yl[4*l + 3] = a3;
        }

        for (short row = 0; row < NR0; ++row) {
            device const uint8_t * q1 = rp[row][i].ql + q_offset_l;
            device const uint8_t * q2 = q1 + 32;
            device const uint8_t * qh = rp[row][i].qh + q_offset_h;
            device const char    * sc = rp[row][i].scales + is;
            device const half    * dh = &rp[row][i].d;
            float4 sums = {0.f, 0.f, 0.f, 0.f};

            FOR_UNROLL (short l = 0; l < 4; ++l) {
                sums[0] += yl[4*l + 0] * ((char)((q1[l] & 0xF) | ((qh[l] & kmask1) << 4)) - 32);
                sums[1] += yl[4*l + 1] * ((char)((q2[l] & 0xF) | ((qh[l] & kmask2) << 2)) - 32);
                sums[2] += yl[4*l + 2] * ((char)((q1[l]  >> 4) | ((qh[l] & kmask3) << 0)) - 32);
                sums[3] += yl[4*l + 3] * ((char)((q2[l]  >> 4) | ((qh[l] & kmask4) >> 2)) - 32);
            }

            sumf[row] += dh[0] * (sums[0] * sc[0] + sums[1] * sc[2] +
                                  sums[2] * sc[4] + sums[3] * sc[6]);
        }
    }

    float scale = 1.f;
    if (NORM) {
        const float sum_squares = simd_sum(sum_sq);
        scale = rms_scale(sum_squares, (uint)nb * QK_K, eps);
        if (!rms_usable(sum_squares, scale)) {
            if (tiisg == 0) {
                raise(faults, layer, BLOOMERY_FAULT_SITE_NORM_QUANT);
                for (short row = 0; row < NR0 && first_row + row < rows; ++row) {
                    y[first_row + row] = NAN;
                }
            }
            return;
        }
    }

    for (short row = 0; row < NR0 && first_row + row < rows; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (NORM) {
            sum_all *= scale;
        }
        if (tiisg == 0) {
            // A row the scaled sum leaves not finite (a gain or a weight that is
            // not finite) raises the norm site and is NaN, not a plausible value.
            if (NORM && !isfinite(sum_all)) {
                raise(faults, args.layer, BLOOMERY_FAULT_SITE_NORM_QUANT);
                y[first_row + row] = NAN;
            } else if (RESID) {
                y[first_row + row] += sum_all;
            } else {
                y[first_row + row] = sum_all;
            }
        }
    }
}

kernel void q6k_matvec(device const block_q6_K * W [[buffer(0)]],
                       device const float      * x [[buffer(1)]],
                       device float            * y [[buffer(2)]],
                       constant MatvecArgs     & args [[buffer(3)]],
                       uint3  tgpig [[threadgroup_position_in_grid]],
                       ushort tiisg [[thread_index_in_simdgroup]],
                       ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const int first_row = simd_first_row(tgpig.x, sgitg);
    q6k_rows<false, false>(W, x, y, args.nblocks, args.eps, args.layer,
                           nullptr, nullptr, first_row, (int)args.rows, tiisg);
}

kernel void q6k_matvec_norm(device const block_q6_K * W [[buffer(0)]],
                            device const float      * x [[buffer(1)]],
                            device float            * y [[buffer(2)]],
                            constant MatvecArgs     & args [[buffer(3)]],
                            device const float      * gain [[buffer(4)]],
                            device atomic_uint      * faults [[buffer(5)]],
                            uint3  tgpig [[threadgroup_position_in_grid]],
                            ushort tiisg [[thread_index_in_simdgroup]],
                            ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const int first_row = simd_first_row(tgpig.x, sgitg);
    q6k_rows<true, false>(W, x, y, args.nblocks, args.eps, args.layer,
                          gain, faults, first_row, (int)args.rows, tiisg);
}

kernel void q6k_matvec_resid(device const block_q6_K * W [[buffer(0)]],
                             device const float      * x [[buffer(1)]],
                             device float            * y [[buffer(2)]],
                             constant MatvecArgs     & args [[buffer(3)]],
                             uint3  tgpig [[threadgroup_position_in_grid]],
                             ushort tiisg [[thread_index_in_simdgroup]],
                             ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const int first_row = simd_first_row(tgpig.x, sgitg);
    q6k_rows<false, true>(W, x, y, args.nblocks, args.eps, args.layer,
                          nullptr, nullptr, first_row, (int)args.rows, tiisg);
}

// The row segments of up to four tensors of one type and one width: the
// threadgroup takes the segment whose first threadgroup it is, and the rows
// are the ones a launch of that segment alone would write.
kernel void q6k_matvec_seg(device const block_q6_K * W [[buffer(0)]],
                            device const float      * x [[buffer(1)]],
                            device float            * y [[buffer(2)]],
                            constant SegArgs        & args [[buffer(3)]],
                            device const float      * gain [[buffer(4)]],
                            device atomic_uint      * faults [[buffer(5)]],
                            uint3  tgpig [[threadgroup_position_in_grid]],
                            ushort tiisg [[thread_index_in_simdgroup]],
                            ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    uint s = 0;
    for (uint i = 0; i < args.nseg; ++i) {
        if (tgpig.x >= args.segs[i].first_group) {
            s = i;
        }
    }
    const uint local = tgpig.x - args.segs[s].first_group;
    const int first_row = simd_first_row(local, sgitg);
    q6k_rows<true, false>(
        W + (ulong)args.segs[s].wblocks,
        x,
        y + args.segs[s].out,
        args.nblocks, args.eps, args.layer,
        gain, faults,
        first_row, (int)args.segs[s].rows, tiisg);
}
