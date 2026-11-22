// The few-row mat-vec family: one weight read multiplied into M = 3..8
// activation rows, or with a second activation fragment into M = 9..16,
// straight from device memory on the matrix units, no threadgroup staging of
// the weights. A4's first (the decode owner of an MLX
// 4-bit folder of a Bf16 scale pair); the entry is
//
//   a4_matvec_mrows_resid     y[m, r] += W[r, :] . x[m, :], in place, M a
//                             launch word (3..8), y rows
//                             `stride` apart, x rows `x_stride` apart
//   a4_matvec_mrows2_resid    the same for M = 9..16: two activation fragments
//                             (TM = 2) against each decoded weight fragment
//
// The structure is MLX's `affine_qmv_mma` (MIT licence; see
// THIRD_PARTY_NOTICES.md; Copyright (c) 2023-2024 Apple Inc.,
// mlx/backend/metal/kernels/quantized.h, ml-explore/mlx PRs #4787 and #4890):
// a SIMD group's 8x8 fragments hold eight weight rows (A: weight row by
// contraction index) against the eight activation rows (B: contraction index
// by activation row), so one fragment multiply carries every activation row
// of the launch, and a weight is read, decoded and used once per K step. What
// differs from MLX's: the weights are this engine's interleaved 36-byte
// groups (eight code words, then the scale and bias words) read as 4-byte
// words, not MLX's separate planes read as 16-byte vectors; the activations
// and the dequantized weights are f32 fragments; the group's bias folds into
// each weight value as `fma(q, scale, bias)` before the multiply; M is a
// launch word with the rows past it zero; only the count of activation
// fragments, TM (one for M <= 8, two for M <= 16), is a template.
//
// Geometry: a threadgroup is WK = 4 SIMD groups and owns BN = 32 weight rows
// (four 8-row fragments). The K loop is split across the SIMD groups: group s
// takes the 128-value blocks s, s + 4, s + 8, ... . In a block the fragments'
// contraction index c is a 16-value chunk of it: weight lane `fn` holds chunks
// fn and fn + 1 (value 4g + i of each, `a = (chunk fn, chunk fn + 1)`),
// activation lane `fm` holds chunk fm's values 4g + i for its two rows, and
// the 16 multiplies g, i = 0..3 of a block cover the 16 values of every chunk.
// A block is two groups; chunk c is group c / 4's words 2 * (c % 4) and
// 2 * (c % 4) + 1.
//
// THE ORDER IS THE GATE, and it is not the 1-row kernel's: row m of an entry
// here equals the 1-row kernel's on that row alone within the f32
// accumulation band, not in bits. The 1-row kernel sums a lane's values
// block by block and folds the group's bias as `bias * sum(x)`; here every
// value carries its bias before the multiply, the fragment multiply sums eight
// chunks in the unit's own order, the SIMD group's blocks add in K order into
// one accumulator, and the four SIMD groups' accumulators add in SIMD group
// order, 0 + 1 + 2 + 3. Both sum the same K products with f32 rounding, so
// the difference is bounded by a few times K * 2^-24 of sum(|w x|); the gate
// states its band as a multiple of that sum.
//
// Row m of x is read only into column m of the fragments, so an activation
// row's NaN or inf stays in its own outputs. A weight row past `rows` reads
// the last row and is not stored. K is a multiple of 128 (an even group
// count); the host refuses the others. This file comes after quant.metal (the
// A4 group and the BF16 pair's widening, FOR_UNROLL) and gemm.metal (frag_mma).

#include <metal_stdlib>

using namespace metal;

#pragma METAL fp contract(off)

#define MROWS_WK 4
#define MROWS_TN 4
#define MROWS_BN (8 * MROWS_TN)

// The launch's argument block, bound by value at slot 3.
struct MatvecMrowsArgs {
    uint rows;     // output rows an activation row takes
    uint nblocks;  // input columns / 64; even
    uint m;        // activation rows, 3..8 (3..16 in the two-fragment entry)
    uint stride;   // floats from one activation row's outputs to the next
    uint x_stride; // floats from one activation row's inputs to the next, a multiple of 4
};

static_assert(sizeof(MatvecMrowsArgs) == 20, "MatvecMrowsArgs is five words");

template <short TM>
kernel void a4_matvec_mrows_resid_t(device const block_mlx_a4 * W [[buffer(0)]],
                                    device const float      * x [[buffer(1)]],
                                    device float            * y [[buffer(2)]],
                                    constant MatvecMrowsArgs & args [[buffer(3)]],
                                    uint3  tgpig [[threadgroup_position_in_grid]],
                                    ushort tiisg [[thread_index_in_simdgroup]],
                                    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    // The lane's place in an 8x8 fragment: elements [fm][fn] and [fm][fn + 1].
    const short qid = (short)(tiisg >> 2);
    const short fm = (qid & 4) + ((tiisg >> 1) & 3);
    const short fn = ((qid & 2) << 1) + ((tiisg & 1) << 1);

    const int nb = (int)args.nblocks;
    const int y_row = (int)tgpig.x * MROWS_BN;
    const int last_row = (int)args.rows - 1;

    // This lane's weight rows (a row past the matrix reads the last), and the
    // group and words of the two chunks it holds in a block.
    device const block_mlx_a4 * wrow[MROWS_TN];
    FOR_UNROLL (short t = 0; t < MROWS_TN; ++t) {
        wrow[t] = W + (ulong)min(y_row + 8 * t + (int)fm, last_row) * (ulong)nb + (fn >> 2);
    }
    const short word0 = (short)(2 * (fn & 3));

    // This lane's two activation rows in each of the TM activation fragments
    // (fragment j holds rows 8j to 8j + 7); a row past M is zero.
    bool live0[TM];
    bool live1[TM];
    device const float * xr0[TM];
    device const float * xr1[TM];
    FOR_UNROLL (short j = 0; j < TM; ++j) {
        live0[j] = (uint)(8 * j + fn) < args.m;
        live1[j] = (uint)(8 * j + fn + 1) < args.m;
        xr0[j] = x + (ulong)(8 * j + fn) * (ulong)args.x_stride + 16 * fm;
        xr1[j] = x + (ulong)(8 * j + fn + 1) * (ulong)args.x_stride + 16 * fm;
    }

    vec<float, 2> C[MROWS_TN][TM];
    FOR_UNROLL (short t = 0; t < MROWS_TN; ++t) {
        FOR_UNROLL (short j = 0; j < TM; ++j) {
            C[t][j] = vec<float, 2>(0.f);
        }
    }

    for (int block = (int)sgitg; 2 * block < nb; block += MROWS_WK) {
        const int k = block * 128;
        float xs0[TM][16];
        float xs1[TM][16];
        FOR_UNROLL (short j = 0; j < TM; ++j) {
            FOR_UNROLL (short h = 0; h < 4; ++h) {
                const float4 a0 = live0[j] ? *(device const float4 *)(xr0[j] + k + 4 * h) : float4(0.f);
                const float4 a1 = live1[j] ? *(device const float4 *)(xr1[j] + k + 4 * h) : float4(0.f);
                xs0[j][4 * h + 0] = a0.x;
                xs0[j][4 * h + 1] = a0.y;
                xs0[j][4 * h + 2] = a0.z;
                xs0[j][4 * h + 3] = a0.w;
                xs1[j][4 * h + 0] = a1.x;
                xs1[j][4 * h + 1] = a1.y;
                xs1[j][4 * h + 2] = a1.z;
                xs1[j][4 * h + 3] = a1.w;
            }
        }
        uint  ws[MROWS_TN][4];
        float sc[MROWS_TN];
        float bs[MROWS_TN];
        FOR_UNROLL (short t = 0; t < MROWS_TN; ++t) {
            device const uint * blk = (device const uint *)(wrow[t] + 2 * block);
            FOR_UNROLL (short l = 0; l < 4; ++l) {
                ws[t][l] = blk[word0 + l];
            }
            const uint sb = blk[8];
            sc[t] = AffineBf16::scale_of_word(sb);
            bs[t] = AffineBf16::bias_of_word(sb);
        }
        FOR_UNROLL (short g = 0; g < 4; ++g) {
            FOR_UNROLL (short i = 0; i < 4; ++i) {
                const short f = 4 * g + i;
                const short shift = (short)(4 * (4 * (g & 1) + i));
                FOR_UNROLL (short t = 0; t < MROWS_TN; ++t) {
                    const uint qx = (ws[t][g >> 1] >> shift) & 0xFu;
                    const uint qy = (ws[t][2 + (g >> 1)] >> shift) & 0xFu;
                    const vec<float, 2> a = vec<float, 2>(fma((float)qx, sc[t], bs[t]),
                                                          fma((float)qy, sc[t], bs[t]));
                    // The weight fragment is decoded once and multiplied
                    // against every activation fragment.
                    FOR_UNROLL (short j = 0; j < TM; ++j) {
                        const vec<float, 2> b = vec<float, 2>(xs0[j][f], xs1[j][f]);
                        frag_mma(C[t][j], a, b, C[t][j]);
                    }
                }
            }
        }
    }

    // The SIMD groups' accumulators add in SIMD group order into group 0's.
    threadgroup float partial[(MROWS_WK - 1) * MROWS_TN * TM * 2 * 32];
    if (sgitg > 0) {
        threadgroup float * p = partial + (sgitg - 1) * MROWS_TN * TM * 2 * 32 + tiisg;
        FOR_UNROLL (short t = 0; t < MROWS_TN; ++t) {
            FOR_UNROLL (short j = 0; j < TM; ++j) {
                p[((TM * t + j) * 2 + 0) * 32] = C[t][j].x;
                p[((TM * t + j) * 2 + 1) * 32] = C[t][j].y;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgitg != 0) {
        return;
    }
    for (short s = 0; s < MROWS_WK - 1; ++s) {
        const threadgroup float * p = partial + s * MROWS_TN * TM * 2 * 32 + tiisg;
        FOR_UNROLL (short t = 0; t < MROWS_TN; ++t) {
            FOR_UNROLL (short j = 0; j < TM; ++j) {
                C[t][j].x += p[((TM * t + j) * 2 + 0) * 32];
                C[t][j].y += p[((TM * t + j) * 2 + 1) * 32];
            }
        }
    }
    FOR_UNROLL (short t = 0; t < MROWS_TN; ++t) {
        const int row = y_row + 8 * t + (int)fm;
        if (row <= last_row) {
            FOR_UNROLL (short j = 0; j < TM; ++j) {
                if (live0[j]) {
                    device float * out = y + (ulong)(8 * j + fn) * (ulong)args.stride + (uint)row;
                    *out = *out + C[t][j].x;
                }
                if (live1[j]) {
                    device float * out = y + (ulong)(8 * j + fn + 1) * (ulong)args.stride + (uint)row;
                    *out = *out + C[t][j].y;
                }
            }
        }
    }
}

// The entries: one activation fragment (M = 3..8) and two (M = 9..16).
#define MROWS_ENTRY(NAME, TM)                                                           \
    template [[host_name(NAME)]]                                                        \
    kernel void a4_matvec_mrows_resid_t<TM>(device const block_mlx_a4 * W [[buffer(0)]],\
                                            device const float * x [[buffer(1)]],       \
                                            device float * y [[buffer(2)]],             \
                                            constant MatvecMrowsArgs & args [[buffer(3)]],\
                                            uint3 tgpig [[threadgroup_position_in_grid]],\
                                            ushort tiisg [[thread_index_in_simdgroup]],  \
                                            ushort sgitg [[simdgroup_index_in_threadgroup]]);

MROWS_ENTRY("a4_matvec_mrows_resid", 1)
MROWS_ENTRY("a4_matvec_mrows2_resid", 2)
