// The dense quantized GEMM: y[r][n] = sum_k W[n][k] * x[r][k] over R
// activation rows, one dispatch. The entry points, under their host names:
//
//   gemm_{q4k,q6k,a4,a8,q8_0,q5k}_{store,resid}[_seg]_{f16,f32,h16}
//   gemm_cast_rows
//
//   store    y = W . x
//   resid    y = y + W . x, in place: the residual epilogue, the decode
//            RESID meaning
//   _seg     the weight rows of up to four tensors of one type and one
//            width, in one dispatch: each segment's rows are columns of the
//            same R-row output, at its own offset
//   f16      the weights dequantized to half and the activations cast to
//            half: the staged half values load into float fragments, so a
//            product of two half values accumulates in f32 exactly
//   f32      both operands f32
//   h16      the f16 arm over half activation rows: gemm_cast_rows ran the
//            staging's checked cast over them once, so the tile stages the
//            same half values the f16 entry's cast would, without the f32
//            loads and the check its every weight tile repeats
//
// The structure is MLX's quantized matmul (MIT licence; see
// THIRD_PARTY_NOTICES.md), the steel form its `qmm_t` runs where no
// cooperative-tensor matmul hardware exists (the M1 and M4 families):
//   mlx/backend/metal/kernels/quantized.h        qmm_t_impl, the K loop
//   mlx/backend/metal/kernels/steel/gemm/mma.h   BlockMMA, MMATile, the
//                                                fragment reinterpret
//   mlx/backend/metal/kernels/steel/gemm/loader.h  the staged row layout
// The tile, the two-barrier K loop, the per-lane fragment loads and the
// serpentine multiply order are theirs. What differs:
//   - the dequant is the quant owner's own `dequant16` (quant.metal), not a
//     free function: the decode has one owner;
//   - the activations are strided rows (x_stride, y_stride), not a
//     contiguous matrix argument block, and load as scalars composed into
//     float4 for the range check — no window alignment is assumed;
//   - the f16 operand's cast is checked: an activation that is not finite or
//     past the half range raises the activation quantizer's fault site and
//     stages NaN, so its row's outputs are NaN — never a silent inf. The
//     f32 arm takes its activations as they are, like the plain mat-vec;
//   - the residual epilogue (MLX's kernel only stores), and the bounded edge
//     store stays per element, not through a threadgroup staging tile.
//
// Geometry: a threadgroup is 128 threads in four SIMD groups, and owns 64
// weight rows (the output's columns) by 64 activation rows (its rows). The
// weight tiles run across the grid's x, its fastest axis (MLX's order): the
// threadgroups in flight together share one activation tile, whose rows
// are then read from memory once while the weight tiles pass, where the
// other order streams every activation row once per weight tile. A
// K step covers 32 values, both arms — the f32 arm's tile takes 18 KB of
// the 32 KB this Mac allows a threadgroup, the f16 arm's 10 KB, so two to
// three f16 threadgroups sit on one core against the f32 arm's one: the
// wider tile the f16 arm's bytes do not leave room for loses more to that
// than its longer steps save. The staged tiles are
// [row][k] rows of the operand type, the pitch `BK + 16 bytes` (steel's
// padding: adjacent rows' same-k elements land off one 16-byte bank), the
// weights at the tile's top and the activations under them. A thread stages
// whole 16-value chunks — one `dequant16` of a weight row, one checked
// float4 run of an activation row — so every staged write is a contiguous
// vector. The multiply is steel's: every lane holds the two elements its
// SIMD group's 8x8 fragments place on it (`[fm][fn]` and `[fm][fn+1]`), the
// A fragments (activations) read two adjacent staged values, the B
// fragments (weights) one value of two adjacent rows. The f32 arm converts
// its staged values into float fragments as it loads them (the activations
// as one packed pair) and multiplies float matrices; the f16 arm keeps its
// fragments half — a packed pair a load — and multiplies them into the
// float accumulator directly, the products of two half values exact in f32
// either way. A SIMD group owns GEMM_FRAGS_A by GEMM_FRAGS_W fragments
// (half of each tile side, in 8-row interleaved blocks — `8 * (sg >> 1)`
// down, `8 * (sg & 1)` across), multiplied in the serpentine order. Edge
// tiles stage the last live row's values (a defined access, the mat-vec's
// row clamp) and store each element under its own row and column bound, so
// no word past the output's last row or column is written; the residual
// epilogue reads, adds and writes back per element.
//
// This file comes after quant.metal and gemm_shared.metal: it reads the
// quant owners, the fault site's table, the tile's geometry, the argument
// blocks and the activations' checked cast (gemm_stage_x).

#include <metal_stdlib>

using namespace metal;

#pragma METAL fp contract(off)

// The 8x8 fragments one SIMD group owns along each side: two SIMD groups
// share a side, in 8-row blocks sixteen rows apart.
#define GEMM_FRAGS_A (GEMM_ACTS / 16)
#define GEMM_FRAGS_W (GEMM_ROWS / 16)

static_assert(GEMM_THREADS == 4 * 32, "four SIMD groups of 32 threads");
static_assert(GEMM_ROWS % 16 == 0 && GEMM_ACTS % 16 == 0,
              "the tile splits into 8x8 fragments, two SIMD groups to a side");

// The float fragment multiply: a `vec<float, 2>` is the 8x8 simdgroup
// matrix's per-lane storage, so the fragment aliases the matrix through
// `thread_elements()` (mma.h's BaseMMAFrag::mma, the reinterpret is MLX's).
static inline void frag_mma(thread vec<float, 2> &       d,
                            thread const vec<float, 2> & a,
                            thread const vec<float, 2> & b,
                            thread const vec<float, 2> & c) {
    simdgroup_matrix<float, 8, 8> am;
    simdgroup_matrix<float, 8, 8> bm;
    simdgroup_matrix<float, 8, 8> dm;
    reinterpret_cast<thread vec<float, 2> &>(am.thread_elements()) = a;
    reinterpret_cast<thread vec<float, 2> &>(bm.thread_elements()) = b;
    reinterpret_cast<thread vec<float, 2> &>(dm.thread_elements()) = c;
    simdgroup_multiply_accumulate(dm, am, bm, dm);
    d = reinterpret_cast<thread vec<float, 2> &>(dm.thread_elements());
}

// The half-operand form of the fragment multiply: the fragments stay half
// and the accumulator is the float one, so the products of two half values
// are exact in the f32 accumulator without a conversion per element.
static inline void frag_mma_half(thread vec<float, 2> &       d,
                                 thread const vec<half, 2> &  a,
                                 thread const vec<half, 2> &  b,
                                 thread const vec<float, 2> & c) {
    simdgroup_matrix<half, 8, 8> am;
    simdgroup_matrix<half, 8, 8> bm;
    simdgroup_matrix<float, 8, 8> dm;
    reinterpret_cast<thread vec<half, 2> &>(am.thread_elements()) = a;
    reinterpret_cast<thread vec<half, 2> &>(bm.thread_elements()) = b;
    reinterpret_cast<thread vec<float, 2> &>(dm.thread_elements()) = c;
    simdgroup_multiply_accumulate(dm, am, bm, dm);
    d = reinterpret_cast<thread vec<float, 2> &>(dm.thread_elements());
}

// One threadgroup's tile of the GEMM: the whole K loop and the store, shared
// by the plain and the segmented entries the way the mat-vec's row loop is
// (`first_row` there, `r0` and `r1` here).
template <typename Q, typename T, typename X, bool RESID>
inline void gemm_tile(threadgroup char        * shmem,
                      device const typename Q::Block * W,
                      device const X          * x,
                      device float            * y,
                      int rows,
                      int cols,
                      int acts,
                      uint x_stride,
                      uint y_stride,
                      uint layer,
                      device atomic_uint * faults,
                      int r0,
                      int r1,
                      ushort tiitg,
                      ushort sgitg,
                      ushort tiisg) {
    constexpr int BK = gemm_bk<T>();
    constexpr int LDA = gemm_lda<T>();
    // A 16-value chunk is the staging unit: one `dequant16`, one checked
    // float4 run. Every thread stages the same count of them per side.
    constexpr int CHUNKS = BK / 16;
    constexpr int A_CHUNKS = GEMM_ROWS * CHUNKS / GEMM_THREADS;
    constexpr int B_CHUNKS = GEMM_ACTS * CHUNKS / GEMM_THREADS;

    // The staged tiles: the weights at the top, the activations under them.
    threadgroup T * sa = (threadgroup T *)shmem;
    threadgroup T * sb = sa + GEMM_ROWS * LDA;

    // The tile's live rows. The staging rows past them clamp to the last
    // live row — a defined access, never stored.
    const short nr0 = (short)min(GEMM_ROWS, rows - r0);
    const short nr1 = (short)min(GEMM_ACTS, acts - r1);

    const uint nb = (uint)cols / (uint)Q::values_per_block();

    // The staging chunks a thread owns do not change from one K step to the
    // next: its weight row (the chunk's row clamps to the tile's last live
    // row) and its activation row, where each sits in the tile, and the
    // offset of the chunk in the step.
    device const typename Q::Block * a_src[A_CHUNKS];
    threadgroup vec<T, 4> * a_dst[A_CHUNKS];
    short a_koff[A_CHUNKS];
    FOR_UNROLL(short a = 0; a < A_CHUNKS; ++a) {
        const short c0 = (short)tiitg + a * GEMM_THREADS;
        const short row = min((short)(c0 / CHUNKS), (short)(nr0 - 1));
        a_koff[a] = (c0 % CHUNKS) * 16;
        a_src[a] = W + (ulong)(r0 + row) * nb;
        a_dst[a] = (threadgroup vec<T, 4> *)(sa + row * LDA + a_koff[a]);
    }
    device const X * b_src[B_CHUNKS];
    threadgroup vec<T, 4> * b_dst[B_CHUNKS];
    FOR_UNROLL(short b = 0; b < B_CHUNKS; ++b) {
        const short c0 = (short)tiitg + b * GEMM_THREADS;
        const short row = min((short)(c0 / CHUNKS), (short)(nr1 - 1));
        const short koff = (c0 % CHUNKS) * 16;
        b_src[b] = x + (ulong)(r1 + row) * x_stride + koff;
        b_dst[b] = (threadgroup vec<T, 4> *)(sb + row * LDA + koff);
    }

    // The lane's place in its SIMD group's 8x8 fragments (mma.h's
    // get_coord): it holds the elements [fm][fn] and [fm][fn+1] of every
    // fragment its group owns. The group's own blocks sit at 8 rows down
    // (`sgitg >> 1`) and 8 columns across (`sgitg & 1`) of the tile, its
    // four fragments by four another 16 apart — the 8-row interleave.
    const short qid = (short)(tiisg >> 2);
    const short fm = (qid & 4) + ((tiisg >> 1) & 3);
    const short fn = ((qid & 2) << 1) + ((tiisg & 1) << 1);
    const short gm = 8 * (sgitg >> 1);
    const short gn = 8 * (sgitg & 1);

    // The lane's staged read rows, hoisted out of the K loop: the A
    // fragments' activation rows (two adjacent k values each) and the B
    // fragments' weight row pairs (one k value of two adjacent rows).
    threadgroup const T * arow[GEMM_FRAGS_A];
    threadgroup const T * brow[GEMM_FRAGS_W];
    FOR_UNROLL(short i = 0; i < GEMM_FRAGS_A; ++i) {
        arow[i] = sb + (gm + 16 * i + fm) * LDA + fn;
    }
    FOR_UNROLL(short j = 0; j < GEMM_FRAGS_W; ++j) {
        brow[j] = sa + (gn + 16 * j + fn) * LDA + fm;
    }

    vec<float, 2> c[GEMM_FRAGS_A][GEMM_FRAGS_W];
    FOR_UNROLL(short i = 0; i < GEMM_FRAGS_A; ++i) {
        FOR_UNROLL(short j = 0; j < GEMM_FRAGS_W; ++j) {
            c[i][j] = vec<float, 2>(0.f);
        }
    }

    // B: whole 16-value runs of the activation rows, each through the
    // checked cast, or as they are when the rows are half (gemm_stage_x).
    auto stage_b = [&](int k0) {
        FOR_UNROLL(short b = 0; b < B_CHUNKS; ++b) {
            device const X * xr = b_src[b] + k0;
            gemm_stage_x<T>(xr, b_dst[b], faults, layer);
        }
    };

    // A: whole 16-value chunks of the weight rows, one `dequant16`
    // each, staged as four contiguous vectors.
    auto stage_a = [&](int k0) {
        FOR_UNROLL(short a = 0; a < A_CHUNKS; ++a) {
            const int pos = k0 + a_koff[a];
            const int ib = pos / Q::values_per_block();
            const short il = (short)((pos / 16) % (Q::values_per_block() / 16));
            matrix<T, 4, 4> reg;
            Q::template dequant16<T>(a_src[a][ib], il, reg);
            FOR_UNROLL(short i = 0; i < 4; ++i) {
                a_dst[a][i] = vec<T, 4>(reg[i][0], reg[i][1], reg[i][2], reg[i][3]);
            }
        }
    };

    for (int k0 = 0; k0 < cols; k0 += BK) {
        // The previous step's fragments are read: the tiles may be written
        // over.
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // The staging order is per arm: the f16 arm stages the activations
        // first and the f32 arm the weights first. The compiler keeps the
        // source order of the two halves, and each arm runs faster in its
        // own.
        if (sizeof(T) == 2) {
            stage_b(k0);
            stage_a(k0);
        } else {
            stage_a(k0);
            stage_b(k0);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // The multiply: per 8 values of K, the lane loads its eight
        // fragments (four of the activations, four of the weight row pairs)
        // and runs its sixteen fragment multiplies in the serpentine order.
        // The f32 arm converts each staged value as it loads; the f16 arm
        // keeps its fragments half and multiplies them into the float
        // accumulator directly (the products of two half values are exact
        // in f32), an A fragment loading as one packed pair. The SIMD-group
        // barriers are MLX's BlockMMA fences: the fragment loads finish
        // before the multiplies start, which the compiler would otherwise
        // interleave.
        for (short kk = 0; kk < BK; kk += 8) {
            simdgroup_barrier(mem_flags::mem_none);
            vec<float, 2> af[GEMM_FRAGS_A];
            vec<float, 2> bf[GEMM_FRAGS_W];
            vec<half, 2> ah[GEMM_FRAGS_A];
            vec<half, 2> bh[GEMM_FRAGS_W];
            if (sizeof(T) == 2) {
                FOR_UNROLL(short i = 0; i < GEMM_FRAGS_A; ++i) {
                    ah[i] = *(threadgroup const vec<half, 2> *)(arow[i] + kk);
                }
                FOR_UNROLL(short j = 0; j < GEMM_FRAGS_W; ++j) {
                    bh[j] = vec<half, 2>(brow[j][kk], (brow[j] + LDA)[kk]);
                }
            } else {
                FOR_UNROLL(short i = 0; i < GEMM_FRAGS_A; ++i) {
                    af[i] = *(threadgroup const vec<float, 2> *)(arow[i] + kk);
                }
                FOR_UNROLL(short j = 0; j < GEMM_FRAGS_W; ++j) {
                    bf[j] = vec<float, 2>(brow[j][kk], (brow[j] + LDA)[kk]);
                }
            }
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL(short m = 0; m < GEMM_FRAGS_A; ++m) {
                FOR_UNROLL(short n = 0; n < GEMM_FRAGS_W; ++n) {
                    const short ns = (m % 2) ? GEMM_FRAGS_W - 1 - n : n;
                    if (sizeof(T) == 2) {
                        frag_mma_half(c[m][ns], ah[m], bh[ns], c[m][ns]);
                    } else {
                        frag_mma(c[m][ns], af[m], bf[ns], c[m][ns]);
                    }
                }
            }
        }
    }

    // The store: a tile the output fully holds writes every element; every
    // other tile (an edge, or the residual epilogue, whose read-add needs
    // the element path) stores each element under its own row and column
    // bound — no word past the output's last row or column is written.
    if (!RESID && nr0 == GEMM_ROWS && nr1 == GEMM_ACTS) {
        FOR_UNROLL(short i = 0; i < GEMM_FRAGS_A; ++i) {
            device float * p = y + (ulong)(r1 + gm + 16 * i + fm) * y_stride + (uint)r0 + gn + fn;
            FOR_UNROLL(short j = 0; j < GEMM_FRAGS_W; ++j) {
                p[16 * j] = c[i][j].x;
                p[16 * j + 1] = c[i][j].y;
            }
        }
    } else {
        FOR_UNROLL(short i = 0; i < GEMM_FRAGS_A; ++i) {
            const short m = gm + 16 * i + fm;
            if (m >= nr1) {
                continue;
            }
            device float * p = y + (ulong)(r1 + m) * y_stride + (uint)r0;
            FOR_UNROLL(short j = 0; j < GEMM_FRAGS_W; ++j) {
                const short n = gn + 16 * j + fn;
                if (n < nr0) {
                    p[n] = RESID ? p[n] + c[i][j].x : c[i][j].x;
                }
                if (n + 1 < nr0) {
                    p[n + 1] = RESID ? p[n + 1] + c[i][j].y : c[i][j].y;
                }
            }
        }
    }
}

// The plain GEMM, storing or adding into the output.
template <typename Q, typename T, typename X, bool RESID>
kernel void gemm_kernel(device const typename Q::Block * W [[buffer(0)]],
                        device const X          * x [[buffer(1)]],
                        device float            * y [[buffer(2)]],
                        constant GemmArgs       & args [[buffer(3)]],
                        device atomic_uint      * faults [[buffer(4)]],
                        uint3  tgpig [[threadgroup_position_in_grid]],
                        ushort tiitg [[thread_index_in_threadgroup]],
                        ushort sgitg [[simdgroup_index_in_threadgroup]],
                        ushort tiisg [[thread_index_in_simdgroup]]) {
    threadgroup char shmem[gemm_shmem_bytes<T>()];
    gemm_tile<Q, T, X, RESID>(shmem, W, x, y, (int)args.rows, (int)args.cols, (int)args.acts,
                           args.x_stride, args.y_stride, args.layer, faults,
                           (int)tgpig.x * GEMM_ROWS, (int)tgpig.y * GEMM_ACTS,
                           tiitg, sgitg, tiisg);
}

// The weight rows of up to four tensors of one type and one width: the
// threadgroup takes the segment whose first weight-tile group it is at or
// past, and the rows are the ones a launch of that segment alone would
// write, at the segment's own column offset.
template <typename Q, typename T, typename X, bool RESID>
kernel void gemm_seg_kernel(device const typename Q::Block * W [[buffer(0)]],
                            device const X          * x [[buffer(1)]],
                            device float            * y [[buffer(2)]],
                            constant GemmSegArgs    & args [[buffer(3)]],
                            device atomic_uint      * faults [[buffer(4)]],
                            uint3  tgpig [[threadgroup_position_in_grid]],
                            ushort tiitg [[thread_index_in_threadgroup]],
                            ushort sgitg [[simdgroup_index_in_threadgroup]],
                            ushort tiisg [[thread_index_in_simdgroup]]) {
    threadgroup char shmem[gemm_shmem_bytes<T>()];
    uint s = 0;
    for (uint i = args.nseg; i > 0; --i) {
        if (tgpig.x >= args.segs[i - 1u].first_group) {
            s = i - 1u;
            break;
        }
    }
    const uint local = tgpig.x - args.segs[s].first_group;
    const ulong wblocks = args.segs[s].wblocks;
    gemm_tile<Q, T, X, RESID>(shmem,
                           W + wblocks,
                           x,
                           y + args.segs[s].out,
                           (int)args.segs[s].rows, (int)args.cols, (int)args.acts,
                           args.x_stride, args.y_stride, args.layer, faults,
                           (int)local * GEMM_ROWS, (int)tgpig.y * GEMM_ACTS,
                           tiitg, sgitg, tiisg);
}

// The entries, under the names the host launches: every weight type, both
// epilogues, both operand types and the f16 arm over half rows, plain and
// segmented.

#define GEMM_ENTRY(SUFFIX, Q, T, RESID, SEG)                                          \
    template [[host_name("gemm_" #SUFFIX)]] kernel void SEG<Q, T, float, RESID>(      \
        device const Q::Block * W [[buffer(0)]],                                     \
        device const float    * x [[buffer(1)]],                                     \
        device float          * y [[buffer(2)]],                                     \
        constant GemmArgs     & args [[buffer(3)]],                                  \
        device atomic_uint    * faults [[buffer(4)]],                                \
        uint3  tgpig [[threadgroup_position_in_grid]],                               \
        ushort tiitg [[thread_index_in_threadgroup]],                                \
        ushort sgitg [[simdgroup_index_in_threadgroup]],                              \
        ushort tiisg [[thread_index_in_simdgroup]]);

#define GEMM_SEG_ENTRY(SUFFIX, Q, T, RESID, SEG)                                      \
    template [[host_name("gemm_" #SUFFIX)]] kernel void SEG<Q, T, float, RESID>(      \
        device const Q::Block * W [[buffer(0)]],                                     \
        device const float    * x [[buffer(1)]],                                     \
        device float          * y [[buffer(2)]],                                     \
        constant GemmSegArgs  & args [[buffer(3)]],                                  \
        device atomic_uint    * faults [[buffer(4)]],                                \
        uint3  tgpig [[threadgroup_position_in_grid]],                               \
        ushort tiitg [[thread_index_in_threadgroup]],                                \
        ushort sgitg [[simdgroup_index_in_threadgroup]],                              \
        ushort tiisg [[thread_index_in_simdgroup]]);

GEMM_ENTRY(q4k_store_f16, Q4K, half, false, gemm_kernel)
GEMM_ENTRY(q4k_store_f32, Q4K, float, false, gemm_kernel)
GEMM_ENTRY(q4k_resid_f16, Q4K, half, true, gemm_kernel)
GEMM_ENTRY(q4k_resid_f32, Q4K, float, true, gemm_kernel)
GEMM_SEG_ENTRY(q4k_seg_store_f16, Q4K, half, false, gemm_seg_kernel)
GEMM_SEG_ENTRY(q4k_seg_store_f32, Q4K, float, false, gemm_seg_kernel)
GEMM_SEG_ENTRY(q4k_seg_resid_f16, Q4K, half, true, gemm_seg_kernel)
GEMM_SEG_ENTRY(q4k_seg_resid_f32, Q4K, float, true, gemm_seg_kernel)
GEMM_ENTRY(q6k_store_f16, Q6K, half, false, gemm_kernel)
GEMM_ENTRY(q6k_store_f32, Q6K, float, false, gemm_kernel)
GEMM_ENTRY(q6k_resid_f16, Q6K, half, true, gemm_kernel)
GEMM_ENTRY(q6k_resid_f32, Q6K, float, true, gemm_kernel)
GEMM_SEG_ENTRY(q6k_seg_store_f16, Q6K, half, false, gemm_seg_kernel)
GEMM_SEG_ENTRY(q6k_seg_store_f32, Q6K, float, false, gemm_seg_kernel)
GEMM_SEG_ENTRY(q6k_seg_resid_f16, Q6K, half, true, gemm_seg_kernel)
GEMM_SEG_ENTRY(q6k_seg_resid_f32, Q6K, float, true, gemm_seg_kernel)
GEMM_ENTRY(a4_store_f16, A4, half, false, gemm_kernel)
GEMM_ENTRY(a4_store_f32, A4, float, false, gemm_kernel)
GEMM_ENTRY(a4_resid_f16, A4, half, true, gemm_kernel)
GEMM_ENTRY(a4_resid_f32, A4, float, true, gemm_kernel)
GEMM_SEG_ENTRY(a4_seg_store_f16, A4, half, false, gemm_seg_kernel)
GEMM_SEG_ENTRY(a4_seg_store_f32, A4, float, false, gemm_seg_kernel)
GEMM_SEG_ENTRY(a4_seg_resid_f16, A4, half, true, gemm_seg_kernel)
GEMM_SEG_ENTRY(a4_seg_resid_f32, A4, float, true, gemm_seg_kernel)
GEMM_ENTRY(a8_store_f16, A8, half, false, gemm_kernel)
GEMM_ENTRY(a8_store_f32, A8, float, false, gemm_kernel)
GEMM_ENTRY(a8_resid_f16, A8, half, true, gemm_kernel)
GEMM_ENTRY(a8_resid_f32, A8, float, true, gemm_kernel)
GEMM_SEG_ENTRY(a8_seg_store_f16, A8, half, false, gemm_seg_kernel)
GEMM_SEG_ENTRY(a8_seg_store_f32, A8, float, false, gemm_seg_kernel)
GEMM_SEG_ENTRY(a8_seg_resid_f16, A8, half, true, gemm_seg_kernel)
GEMM_SEG_ENTRY(a8_seg_resid_f32, A8, float, true, gemm_seg_kernel)
GEMM_ENTRY(q8_0_store_f16, Q8_0, half, false, gemm_kernel)
GEMM_ENTRY(q8_0_store_f32, Q8_0, float, false, gemm_kernel)
GEMM_ENTRY(q8_0_resid_f16, Q8_0, half, true, gemm_kernel)
GEMM_ENTRY(q8_0_resid_f32, Q8_0, float, true, gemm_kernel)
GEMM_SEG_ENTRY(q8_0_seg_store_f16, Q8_0, half, false, gemm_seg_kernel)
GEMM_SEG_ENTRY(q8_0_seg_store_f32, Q8_0, float, false, gemm_seg_kernel)
GEMM_SEG_ENTRY(q8_0_seg_resid_f16, Q8_0, half, true, gemm_seg_kernel)
GEMM_SEG_ENTRY(q8_0_seg_resid_f32, Q8_0, float, true, gemm_seg_kernel)
GEMM_ENTRY(q5k_store_f16, Q5K, half, false, gemm_kernel)
GEMM_ENTRY(q5k_store_f32, Q5K, float, false, gemm_kernel)
GEMM_ENTRY(q5k_resid_f16, Q5K, half, true, gemm_kernel)
GEMM_ENTRY(q5k_resid_f32, Q5K, float, true, gemm_kernel)
GEMM_SEG_ENTRY(q5k_seg_store_f16, Q5K, half, false, gemm_seg_kernel)
GEMM_SEG_ENTRY(q5k_seg_store_f32, Q5K, float, false, gemm_seg_kernel)
GEMM_SEG_ENTRY(q5k_seg_resid_f16, Q5K, half, true, gemm_seg_kernel)
GEMM_SEG_ENTRY(q5k_seg_resid_f32, Q5K, float, true, gemm_seg_kernel)

#define GEMM_H16_ENTRY(SUFFIX, Q, RESID, SEG, ARGS)                                   \
    template [[host_name("gemm_" #SUFFIX)]] kernel void SEG<Q, half, half, RESID>(    \
        device const Q::Block * W [[buffer(0)]],                                     \
        device const half     * x [[buffer(1)]],                                     \
        device float          * y [[buffer(2)]],                                     \
        constant ARGS         & args [[buffer(3)]],                                  \
        device atomic_uint    * faults [[buffer(4)]],                                \
        uint3  tgpig [[threadgroup_position_in_grid]],                               \
        ushort tiitg [[thread_index_in_threadgroup]],                                \
        ushort sgitg [[simdgroup_index_in_threadgroup]],                              \
        ushort tiisg [[thread_index_in_simdgroup]]);

#define GEMM_H16_ENTRIES(STEM, Q)                                                     \
    GEMM_H16_ENTRY(STEM##_store_h16, Q, false, gemm_kernel, GemmArgs)                 \
    GEMM_H16_ENTRY(STEM##_resid_h16, Q, true, gemm_kernel, GemmArgs)                  \
    GEMM_H16_ENTRY(STEM##_seg_store_h16, Q, false, gemm_seg_kernel, GemmSegArgs)      \
    GEMM_H16_ENTRY(STEM##_seg_resid_h16, Q, true, gemm_seg_kernel, GemmSegArgs)

GEMM_H16_ENTRIES(q4k, Q4K)
GEMM_H16_ENTRIES(q6k, Q6K)
GEMM_H16_ENTRIES(a4, A4)
GEMM_H16_ENTRIES(a8, A8)
GEMM_H16_ENTRIES(q8_0, Q8_0)
GEMM_H16_ENTRIES(q5k, Q5K)

// The argument block of the activations' cast, bound by value at slot 2.
struct GemmCastArgs {
    uint cols;     // the values of a row, a multiple of four
    uint acts;     // the rows
    uint x_stride; // the f32 elements between x's rows, at least cols
    uint y_stride; // the half elements between y's rows, a multiple of four
    uint layer;    // the layer a raise is recorded in
};

static_assert(sizeof(GemmCastArgs) == 20, "GemmCastArgs is five words");

// The threads of a cast threadgroup, four values each: 512 values of one row.
#define GEMM_CAST_THREADS 128

// The half rows the _h16 entries read: every value of `acts` rows of `cols`
// through the staging's own checked cast (gemm_half4), once — the f16
// entries run it over every value once per weight tile. A thread loads four
// values as scalars (no window alignment is assumed, the staging's own
// loads) and stores them as one half4; the grid is the rows' 512-value runs
// across x and the rows down y.
kernel void gemm_cast_rows(device const float      * x [[buffer(0)]],
                           device half             * y [[buffer(1)]],
                           constant GemmCastArgs   & args [[buffer(2)]],
                           device atomic_uint      * faults [[buffer(3)]],
                           uint3  tgpig [[threadgroup_position_in_grid]],
                           ushort tiitg [[thread_index_in_threadgroup]]) {
    const uint c = 4u * (tgpig.x * GEMM_CAST_THREADS + (uint)tiitg);
    if (c >= args.cols || tgpig.y >= args.acts) {
        return;
    }
    device const float * xr = x + (ulong)tgpig.y * args.x_stride + c;
    const float4 v = float4(xr[0], xr[1], xr[2], xr[3]);
    *(device half4 *)(y + (ulong)tgpig.y * args.y_stride + c) = gemm_half4(v, faults, args.layer);
}
