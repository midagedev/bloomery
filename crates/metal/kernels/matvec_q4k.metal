// The Q4_K mat-vec: y[r] = sum_c W[r, c] * x[c] for one f32 activation vector,
// W in GGUF block_q4_K layout, one dispatch. The entries:
//
//   q4k_matvec                y = W . x
//   q4k_matvec_norm           y = W . rms_norm(x, gain, eps): the norm applied
//                             as x is loaded (norm.metal)
//   q4k_matvec_resid          y = y + W . x, in place: the residual epilogue
//   q4k_matvec_seg            the row segments of up to four tensors of one
//                             type and one width, in one dispatch
//   q4k_matvec_headnorm_resid y = y + W . (g * head_scale): the gated delta
//                             net's out projection, the head scale applied to
//                             each block's head groups
//
// Adapted from llama.cpp's kernel_mul_mv_q4_K_f32 (MIT licence; see
// THIRD_PARTY_NOTICES.md):
//   ggml/src/ggml-metal/kernels/mul_mv.metal
// The lane map (ix/it/iq/ir), the four-block loop, the uint16 nibble trick, the
// row loop over a SIMD group's rows and the simd_sum reduction are llama.cpp's:
// 2 rows per SIMD group. The SIMD groups in a threadgroup are the function
// constant of norm.metal: 2 for the dense mat-vec, 8 for the output head.
//
// What differs from llama.cpp's loop:
//   - the matrix is not padded: a SIMD group past the last row returns, and each
//     row's blocks are addressed from a pointer taken once before the loop,
//     clamped to the last row, so a row past the last re-reads the last row (a
//     defined access) and is never stored; llama.cpp advances one pointer by a
//     row's bytes and reads whole threadgroups of rows past the end;
//   - the variants above.
//
// This file comes after norm.metal and defines what the Q6_K and embed files
// use of it: QK_K, FOR_UNROLL and block_q4_K.

#include <metal_stdlib>

using namespace metal;

#pragma METAL fp contract(off)

#define QK_K 256
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

struct block_q4_K {
    half    d;
    half    dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};

static_assert(sizeof(block_q4_K) == 144, "block_q4_K is 144 bytes");
static_assert(BLOOMERY_MATVEC_ROWS_PER_SIMD_GROUP == 2,
              "the tail step below is the pair's: it is written for two rows");

// The first row a SIMD group owns. `tg` is the threadgroup's index in the
// grid, or its index inside a segment.
inline int simd_first_row(uint tg, ushort sg) {
    return (int)((tg * BLOOMERY_MATVEC_SIMD_GROUPS + sg) * BLOOMERY_MATVEC_ROWS_PER_SIMD_GROUP);
}

// The rows of one segment or one tensor a threadgroup owns. The norm scales
// the whole input at the end, the head norm scales each block's two head
// groups once at the end of the block, and the residual adds into the
// output: the dot product is the same in all of them, so the rows they write
// are the same bits.
template <bool NORM, bool RESID, bool HEADNORM>
inline void q4k_rows(device const block_q4_K * W,
                      device const float      * x,
                      device float            * y,
                      int nblocks,
                      float eps,
                      uint layer,
                      device const float      * gain,
                      threadgroup const float  * scales,
                      device atomic_uint      * faults,
                      int first_row,
                      int rows,
                      ushort tiisg) {
    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;
    constexpr short NR0 = BLOOMERY_MATVEC_ROWS_PER_SIMD_GROUP;

    const short ix = tiisg/8;  // 0...3
    const short it = tiisg%8;  // 0...7
    const short iq = it/4;     // 0 or 1
    const short ir = it%4;     // 0...3

    const int nb = nblocks;
    if (first_row >= rows) {
        return;
    }
    // Each row's blocks, the last row's for a row past the last.
    device const block_q4_K * rp[NR0];
    for (short r = 0; r < NR0; ++r) {
        rp[r] = W + (ulong)min(first_row + r, rows - 1) * (ulong)nb;
    }

    float yl[16];
    float yh[16];
    float sumf[NR0];
    for (short r = 0; r < NR0; ++r) sumf[r] = 0.f;
    float sum_sq = 0.f;

    const int y_offset = ix * QK_K + 64 * iq + 8 * ir;
    device const float * y4 = x + y_offset;
    // The gain is bound by the norm entry only: the others form no pointer
    // from it.
    device const float * g4 = NORM ? gain + y_offset : gain;

    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

    for (int ib = ix; ib < nb; ib += 4) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        // The block's two heads: the head scale of each, once for the block.
        float lo = 1.f;
        float hi = 1.f;
        if (HEADNORM) {
            lo = scales[2*ib];
            hi = scales[2*ib+1];
        }
        for (short i = 0; i < 8; ++i) {
            float a0 = y4[i+  0];
            float a1 = y4[i+ 32];
            float a2 = y4[i+128];
            float a3 = y4[i+160];
            if (NORM) {
                sum_sq += (a0 * a0 + a1 * a1) + (a2 * a2 + a3 * a3);
                a0 *= g4[i+  0];
                a1 *= g4[i+ 32];
                a2 *= g4[i+128];
                a3 *= g4[i+160];
            }
            yl[i+0] = a0; sumy[0] += a0;
            yl[i+8] = a1; sumy[1] += a1;
            yh[i+0] = a2; sumy[2] += a2;
            yh[i+8] = a3; sumy[3] += a3;
        }
        // A value the gated numerator cannot carry, or a group sum that left
        // the finite range: the gate's site is raised, and the sums it is in
        // are not finite either, so the rows are NaN.
        if (HEADNORM && (!isfinite(sumy[0]) || !isfinite(sumy[1]) ||
                         !isfinite(sumy[2]) || !isfinite(sumy[3]))) {
            raise(faults, layer, BLOOMERY_FAULT_SITE_LINEAR_GATE);
        }

        for (short row = 0; row < NR0; ++row) {
            device const uint16_t * sc = (device const uint16_t *)rp[row][ib].scales + iq;
            device const uint16_t * q1 = (device const uint16_t *)rp[row][ib].qs + 16 * iq + 4 * ir;
            device const half     * dh = &rp[row][ib].d;

            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

            device const uint16_t * q2 = q1 + 32;

            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};

            FOR_UNROLL (short i = 0; i < 4; ++i) {
                acc1[0] += yl[2*i + 0] * (q1[i] & 0x000F);
                acc1[1] += yl[2*i + 1] * (q1[i] & 0x0F00);
                acc1[2] += yl[2*i + 8] * (q1[i] & 0x00F0);
                acc1[3] += yl[2*i + 9] * (q1[i] & 0xF000);
                acc2[0] += yh[2*i + 0] * (q2[i] & 0x000F);
                acc2[1] += yh[2*i + 1] * (q2[i] & 0x0F00);
                acc2[2] += yh[2*i + 8] * (q2[i] & 0x00F0);
                acc2[3] += yh[2*i + 9] * (q2[i] & 0xF000);
            }

            // The head scale multiplies each group's whole term once, not
            // every value in it: the dot is linear in the input.
            if (HEADNORM) {
                sumf[row] += lo * (dh[0] * ((acc1[0] + 1.f/256.f * acc1[1]) * sc8[0] +
                                            (acc1[2] + 1.f/256.f * acc1[3]) * sc8[1] * 1.f/16.f) -
                                   dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3])) +
                             hi * (dh[0] * ((acc2[0] + 1.f/256.f * acc2[1]) * sc8[4] +
                                            (acc2[2] + 1.f/256.f * acc2[3]) * sc8[5] * 1.f/16.f) -
                                   dh[1] * (sumy[2] * sc8[6] + sumy[3] * sc8[7]));
            } else {
                sumf[row] += dh[0] * ((acc1[0] + 1.f/256.f * acc1[1]) * sc8[0] +
                                      (acc1[2] + 1.f/256.f * acc1[3]) * sc8[1] * 1.f/16.f +
                                      (acc2[0] + 1.f/256.f * acc2[1]) * sc8[4] +
                                      (acc2[2] + 1.f/256.f * acc2[3]) * sc8[5] * 1.f/16.f) -
                             dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                                      sumy[2] * sc8[6] + sumy[3] * sc8[7]);
            }
        }

        y4 += 4 * QK_K;
        if (NORM) {
            g4 += 4 * QK_K;
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

kernel void q4k_matvec(device const block_q4_K * W [[buffer(0)]],
                       device const float      * x [[buffer(1)]],
                       device float            * y [[buffer(2)]],
                       constant MatvecArgs     & args [[buffer(3)]],
                       uint3  tgpig [[threadgroup_position_in_grid]],
                       ushort tiisg [[thread_index_in_simdgroup]],
                       ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const int first_row = simd_first_row(tgpig.x, sgitg);
    q4k_rows<false, false, false>(W, x, y, args.nblocks, args.eps, args.layer,
                                  nullptr, nullptr, nullptr, first_row, (int)args.rows, tiisg);
}

kernel void q4k_matvec_norm(device const block_q4_K * W [[buffer(0)]],
                            device const float      * x [[buffer(1)]],
                            device float            * y [[buffer(2)]],
                            constant MatvecArgs     & args [[buffer(3)]],
                            device const float      * gain [[buffer(4)]],
                            device atomic_uint      * faults [[buffer(5)]],
                            uint3  tgpig [[threadgroup_position_in_grid]],
                            ushort tiisg [[thread_index_in_simdgroup]],
                            ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const int first_row = simd_first_row(tgpig.x, sgitg);
    q4k_rows<true, false, false>(W, x, y, args.nblocks, args.eps, args.layer,
                                 gain, nullptr, faults, first_row, (int)args.rows, tiisg);
}

kernel void q4k_matvec_resid(device const block_q4_K * W [[buffer(0)]],
                             device const float      * x [[buffer(1)]],
                             device float            * y [[buffer(2)]],
                             constant MatvecArgs     & args [[buffer(3)]],
                             uint3  tgpig [[threadgroup_position_in_grid]],
                             ushort tiisg [[thread_index_in_simdgroup]],
                             ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const int first_row = simd_first_row(tgpig.x, sgitg);
    q4k_rows<false, true, false>(W, x, y, args.nblocks, args.eps, args.layer,
                                 nullptr, nullptr, nullptr, first_row, (int)args.rows, tiisg);
}

// The row segments of up to four tensors of one type and one width: the
// threadgroup takes the segment whose first threadgroup it is, and the rows
// are the ones a launch of that segment alone would write.
kernel void q4k_matvec_seg(device const block_q4_K * W [[buffer(0)]],
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
    q4k_rows<true, false, false>(
        W + (ulong)args.segs[s].wblocks,
        x,
        y + args.segs[s].out,
        args.nblocks, args.eps, args.layer,
        gain, nullptr, faults,
        first_row, (int)args.segs[s].rows, tiisg);
}

// The gated delta net's out projection: the input is the gated numerator g
// times the norm scale of the head each column belongs to, and the scale comes
// from the head's partial sums of the squares of its values. The scales are
// computed once per threadgroup; the row loop reads g from device memory the
// way the plain residual path reads its input, and multiplies each block's two
// head groups by their scale once, so the per-value work the plain path does
// not do is two threadgroup loads and two products a block.
kernel void q4k_matvec_headnorm_resid(device const block_q4_K * W [[buffer(0)]],
                                      device const float      * g [[buffer(1)]],
                                      device const float      * partials [[buffer(2)]],
                                      device float            * y [[buffer(3)]],
                                      constant MatvecArgs     & args [[buffer(4)]],
                                      device atomic_uint      * faults [[buffer(5)]],
                                      uint3  tgpig [[threadgroup_position_in_grid]],
                                      ushort tiisg [[thread_index_in_simdgroup]],
                                      ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    // The head scales, in threadgroup memory: a lane reads the scale of each
    // head its columns are in, and the lanes of a SIMD group do not read the
    // same head, so a shuffle cannot share them.
    threadgroup float scales[BLOOMERY_DELTA_VALUE_HEADS];
    if (sgitg == 0 && tiisg < BLOOMERY_DELTA_VALUE_HEADS) {
        const uint head = tiisg;
        float sum = 0.f;
        for (short j = 0; j < BLOOMERY_DELTA_COLUMN_BLOCKS; ++j) {
            sum += partials[head * BLOOMERY_DELTA_COLUMN_BLOCKS + j];
        }
        const float mean = sum / float(BLOOMERY_DELTA_HEAD_COLS);
        const float s = rsqrt(mean + args.eps);
        if (!isfinite(sum) || !isfinite(s)) {
            // A head the norm cannot take is NaN, not a zero head: its values
            // are then not finite and the rows that read them are NaN.
            raise(faults, args.layer, BLOOMERY_FAULT_SITE_LINEAR_GATE);
            scales[head] = NAN;
        } else {
            scales[head] = s;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int first_row = simd_first_row(tgpig.x, sgitg);
    q4k_rows<false, true, true>(W, g, y, args.nblocks, args.eps, args.layer,
                                nullptr, scales, faults, first_row, (int)args.rows, tiisg);
}
