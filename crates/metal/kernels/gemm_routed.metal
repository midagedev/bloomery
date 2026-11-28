// The grouped GEMM of the routed experts, and the dense GEMM over f32 weight
// rows. The entry points, under their host names:
//
//   gemm_routed_{q4k,q6k,a4,a8,a4h,a8h,q8_0,q5k}_{pair,plain}_{f16,f32}
//   gemm_routed_{q4k,q5k}_apart_q8_0_pair_{f16,f32}
//   gemm_routed_{q5k,q6k}_apart_q8_0_plain_{f16,f32}
//   gemm_f32w_store_f32
//
//   pair     y[p][o] = silu(gate_e[o] . x) * (up_e[o] . x) * w[p], the
//            SwiGLU pair of the experts' gate and up, the route weight
//            applied to the activation as the decode fold applies it
//   plain    y[p][n] = down_e[n] . x[p]
//   f32w     y = W . x over f32 weight rows (the router's), f32 operands
//
// The tile body is dense GEMM's (gemm.metal: the K loop, the staged rows, the
// fragment loads and the serpentine multiply are its helpers' and its own
// constants'); what differs is the tile's place. A threadgroup is one tile of
// the route table (route.metal): the group (a routed expert, or the shared
// expert after them), the group's run of places in the expert-sorted slot
// list, and the run's length. The weight rows come from the group's tensors:
// the stack's expert, or the shared tensors for the table's last group. The
// activation rows are the run's places: gathered through the sorted slot
// list to the pass's row (gate and up, whose input is the normed residual of
// the slot's row), or the places' own rows (down, whose input is the pair's
// output at the place). Outputs are written at the place.
//
// Geometry: 128 threads, four SIMD groups; 64 weight rows by ROUTED_ACTS (64)
// activation rows, a K step of 32. SIMD group g multiplies weight rows 16 g
// to 16 g + 15 against every live 8-row fragment of activation rows. The
// pair's 64 weight rows are 32 outputs of the gate and the same 32 of the
// up, staged by SIMD group (rows 16 g on: the gate of outputs 8 g to 8 g + 7,
// then their up), so the lane that holds an output's gate accumulator holds
// its up accumulator, and the epilogue is the lane's own. A group holds
// about 8 U / 256 rows at U rows a pass, often fewer than the tile: the K
// loop is chosen once a tile by its fragment count, ceil(run / 8), and
// stages and multiplies only those fragments. An output's sum order is the
// K loop's alone; the tile's height and its split over the SIMD groups do
// not move it.
//
// Every entry is `gemm_routed_apart_kernel<Q, QS>`, the kernel over two quants,
// the stacks' and the shared expert's: the tile is one group's, so a tile of
// the shared group runs the shared quant's body whole and every other tile
// the stacks'. A file whose shared expert has the stacks' type instantiates it
// with `QS == Q`, under the plain host names; a shared expert of another type
// (`Apart`) takes the `*_apart_*` entries. Those are in this program only: the
// tensor-op program has none, and the host builds them from this one.
//
// This file comes after gemm_shared.metal (the routed tile's geometry, its
// argument block, pair_swiglu and gemm_stage_x), gemm.metal (the helpers
// and the instantiated tile pieces) and route.metal (the table's shape).

#include <metal_stdlib>

using namespace metal;

#pragma METAL fp contract(off)

// F32W: rows of f32 weights, 16 values a block (quant.metal's block_f32_16).
// The GEMM's weight type for the router, whose rows are f32 in the arena.
struct F32W {
    typedef block_f32_16 Block;

    static inline int values_per_block() {
        return 16;
    }

    template <typename T>
    static inline void dequant16(device const Block & blk,
                                 short il,
                                 thread matrix<T, 4, 4> & reg) {
        FOR_UNROLL(short i = 0; i < 16; ++i) {
            reg[i / 4][i % 4] = (T)blk.v[i];
        }
    }
};

GEMM_ENTRY(f32w_store_f32, F32W, float, false, gemm_kernel)

#define ROUTED_FRAGS_A (ROUTED_ACTS / 8)
#define ROUTED_FRAGS_W 2  // a SIMD group's 8-row weight fragments: its 16 weight rows

static_assert(ROUTED_ACTS == ROUTE_TILE, "a tile holds a table tile's slots");
static_assert(ROUTED_FRAGS_A == 8, "the store switch has a case for each A fragment count");
static_assert(ROUTED_ROWS * 2 == GEMM_THREADS && ROUTED_ACTS * 2 == GEMM_THREADS,
              "a thread stages one 16-value chunk of a weight row and of an activation row");
static_assert(ROUTED_ROWS == 4 * 8 * ROUTED_FRAGS_W,
              "the four SIMD groups split the weight rows, 16 rows each");

template <typename T>
constexpr int routed_shmem_bytes() {
    return (ROUTED_ROWS + ROUTED_ACTS) * gemm_lda<T>() * (int)sizeof(T);
}

// One tile's K loop and store over its first NA 8-row A fragments: the rows
// under 8 NA, which hold the run. Every SIMD group multiplies the same NA
// fragments against its own 16 weight rows, so every fragment the loop holds
// multiplies, with no test inside the loop (a test a fragment costs more
// than the multiplies it skips); the rows of the last fragment past the
// run's length multiply the clamped row's values and are not stored. Only
// the rows under 8 NA are staged.
template <typename Q, typename T, bool PAIR, short NA>
inline void routed_tile(threadgroup T                  * sa,
                        threadgroup T                  * sb,
                        device const typename Q::Block * a_src,
                        threadgroup vec<T, 4>          * a_dst,
                        short                            a_koff,
                        device const float             * b_src,
                        threadgroup vec<T, 4>          * b_dst,
                        short                            count,
                        uint                             first,
                        uint                             r0,
                        device float                   * y,
                        device const uint              * sorted,
                        device const float             * route_w,
                        constant RoutedArgs            & args,
                        device atomic_uint             * faults,
                        ushort                           tiitg,
                        ushort                           sgitg,
                        ushort                           tiisg) {
    constexpr int BK = gemm_bk<T>();
    constexpr int LDA = gemm_lda<T>();
    const bool b_stage = (short)(tiitg / 2) < 8 * NA;

    // The lane's place in its 8x8 fragments (gemm.metal's), the group's
    // weight rows 16 sgitg onward, A fragment i the activation rows 8 i on.
    const short qid = (short)(tiisg >> 2);
    const short fm = (qid & 4) + ((tiisg >> 1) & 3);
    const short fn = ((qid & 2) << 1) + ((tiisg & 1) << 1);
    const short gn = 16 * (short)sgitg;

    threadgroup const T * arow[NA];
    threadgroup const T * brow[ROUTED_FRAGS_W];
    FOR_UNROLL(short i = 0; i < NA; ++i) {
        arow[i] = sb + (8 * i + fm) * LDA + fn;
    }
    FOR_UNROLL(short j = 0; j < ROUTED_FRAGS_W; ++j) {
        brow[j] = sa + (gn + 8 * j + fn) * LDA + fm;
    }

    vec<float, 2> c[NA][ROUTED_FRAGS_W];
    FOR_UNROLL(short i = 0; i < NA; ++i) {
        FOR_UNROLL(short j = 0; j < ROUTED_FRAGS_W; ++j) {
            c[i][j] = vec<float, 2>(0.f);
        }
    }

    // One K step's staging into the tile at `at` elements past the shared
    // memory's start: the activation rows the loop multiplies, then the
    // thread's weight chunk.
    auto stage = [&](int k0, int at) {
        if (b_stage) {
            device const float * xr = b_src + k0;
            threadgroup vec<T, 4> * bd = (threadgroup vec<T, 4> *)((threadgroup T *)b_dst + at);
            gemm_stage_x<T>(xr, bd, faults, args.layer);
        }
        const int pos = k0 + a_koff;
        const int ib = pos / Q::values_per_block();
        const short il = (short)((pos / 16) % (Q::values_per_block() / 16));
        matrix<T, 4, 4> reg;
        Q::template dequant16<T>(a_src[ib], il, reg);
        threadgroup vec<T, 4> * ad = (threadgroup vec<T, 4> *)((threadgroup T *)a_dst + at);
        FOR_UNROLL(short i = 0; i < 4; ++i) {
            ad[i] = vec<T, 4>(reg[i][0], reg[i][1], reg[i][2], reg[i][3]);
        }
    };

    // One K step's multiply over the tile at `at`: per 8 values of K the
    // lane loads its fragments and runs the multiplies in the serpentine
    // order, between MLX's two SIMD-group fences.
    auto multiply = [&](int at) {
        for (short kk = 0; kk < BK; kk += 8) {
            simdgroup_barrier(mem_flags::mem_none);
            vec<float, 2> af[NA];
            vec<float, 2> bf[ROUTED_FRAGS_W];
            vec<half, 2> ah[NA];
            vec<half, 2> bh[ROUTED_FRAGS_W];
            if (sizeof(T) == 2) {
                FOR_UNROLL(short i = 0; i < NA; ++i) {
                    ah[i] = *(threadgroup const vec<half, 2> *)(arow[i] + at + kk);
                }
                FOR_UNROLL(short j = 0; j < ROUTED_FRAGS_W; ++j) {
                    threadgroup const T *w = brow[j] + at;
                    bh[j] = vec<half, 2>(w[kk], w[LDA + kk]);
                }
            } else {
                FOR_UNROLL(short i = 0; i < NA; ++i) {
                    af[i] = *(threadgroup const vec<float, 2> *)(arow[i] + at + kk);
                }
                FOR_UNROLL(short j = 0; j < ROUTED_FRAGS_W; ++j) {
                    threadgroup const T *w = brow[j] + at;
                    bf[j] = vec<float, 2>(w[kk], w[LDA + kk]);
                }
            }
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL(short m = 0; m < NA; ++m) {
                FOR_UNROLL(short n = 0; n < ROUTED_FRAGS_W; ++n) {
                    const short ns = (m % 2) ? ROUTED_FRAGS_W - 1 - n : n;
                    if (sizeof(T) == 2) {
                        frag_mma_half(c[m][ns], ah[m], bh[ns], c[m][ns]);
                    } else {
                        frag_mma(c[m][ns], af[m], bf[ns], c[m][ns]);
                    }
                }
            }
        }
    };

    for (int k0 = 0; k0 < (int)args.cols; k0 += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        stage(k0, 0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        multiply(0);
    }

    // The store, per element under the run's length. A pair's group holds
    // the gate of outputs 8 sgitg onward in its first weight fragment and
    // their up in its second, so a lane holds an output's gate and up.
    FOR_UNROLL(short i = 0; i < NA; ++i) {
        const short m = 8 * i + fm;
        if (m >= count) {
            continue;
        }
        const uint place = first + (uint)m;
        device float * p = y + (ulong)place * args.y_stride;
        if (PAIR) {
            const float rw = route_w[sorted[place]];
            const uint o = r0 + (uint)(8 * sgitg + fn);
            p[o] = pair_swiglu(c[i][0].x, c[i][1].x) * rw;
            p[o + 1] = pair_swiglu(c[i][0].y, c[i][1].y) * rw;
        } else {
            FOR_UNROLL(short j = 0; j < ROUTED_FRAGS_W; ++j) {
                const uint n = r0 + (uint)(gn + 8 * j + fn);
                p[n] = c[i][j].x;
                p[n + 1] = c[i][j].y;
            }
        }
    }
}

// A tile's staging and multiply over quant Q, from the group's gate (and up)
// tensors: the kernel's body after the group's weights are picked, taken for
// the stacks' quant and for the shared tensors' quant.
template <typename Q, typename T, bool PAIR>
inline void routed_run(threadgroup char                * shmem,
                       device const typename Q::Block  * wg,
                       device const typename Q::Block  * wu,
                       uint                              tile_y,
                       uint                              first,
                       short                             count,
                       device const float              * x,
                       device float                    * y,
                       device const uint               * sorted,
                       device const float              * route_w,
                       constant RoutedArgs             & args,
                       device atomic_uint              * faults,
                       ushort                            tiitg,
                       ushort                            sgitg,
                       ushort                            tiisg) {
    constexpr int LDA = gemm_lda<T>();
    // The tile's first output: the pair's tile is 32 outputs (64 weight rows
    // of two tensors), the plain tile 64 rows of one.
    const uint r0 = tile_y * (PAIR ? ROUTED_ROWS / 2 : ROUTED_ROWS);

    const uint nb = args.cols / (uint)Q::values_per_block();

    threadgroup T * sa = (threadgroup T *)shmem;
    threadgroup T * sb = sa + ROUTED_ROWS * LDA;

    // A: a thread stages one 16-value chunk of a weight row a K step. The
    // pair's staged rows go by SIMD group: 16 g onward the gate of outputs
    // 8 g to 8 g + 7, then their up.
    const short a_row = (short)(tiitg / 2);
    const short a_koff = (short)((tiitg % 2) * 16);
    device const typename Q::Block * a_src;
    if (PAIR) {
        const uint o = r0 + (uint)(8 * (a_row / 16) + a_row % 8);
        a_src = ((a_row / 8) % 2 == 0 ? wg : wu) + (ulong)o * nb;
    } else {
        a_src = wg + (ulong)(r0 + a_row) * nb;
    }
    threadgroup vec<T, 4> * a_dst = (threadgroup vec<T, 4> *)(sa + a_row * LDA + a_koff);

    // B: a thread stages one 16-value chunk of an activation row; a row past
    // the run's length takes the run's last (a defined access).
    const short b_row = (short)(tiitg / 2);
    const uint b_place = first + (uint)min(b_row, (short)(count - 1));
    const ulong src_row =
        args.gather != 0u ? (ulong)(sorted[b_place] / BLOOMERY_EXPERT_SLOTS) : (ulong)b_place;
    device const float * b_src = x + src_row * args.x_stride + a_koff;
    threadgroup vec<T, 4> * b_dst = (threadgroup vec<T, 4> *)(sb + b_row * LDA + a_koff);

    // The tile's A fragments: the run's rows in 8-row steps, one branch a
    // tile.
    switch ((count + 7) / 8) {
#define ROUTED_TILE_CASE(NA)                                                              \
    case NA:                                                                              \
        routed_tile<Q, T, PAIR, NA>(sa, sb, a_src, a_dst, a_koff, b_src, b_dst,           \
                                    count, first, r0, y, sorted, route_w, args, faults,   \
                                    tiitg, sgitg, tiisg);                                 \
        break;
        ROUTED_TILE_CASE(1)
        ROUTED_TILE_CASE(2)
        ROUTED_TILE_CASE(3)
        ROUTED_TILE_CASE(4)
        ROUTED_TILE_CASE(5)
        ROUTED_TILE_CASE(6)
        ROUTED_TILE_CASE(7)
        ROUTED_TILE_CASE(8)
#undef ROUTED_TILE_CASE
    }
}

template <typename Q, typename QS, typename T, bool PAIR>
kernel void gemm_routed_apart_kernel(device const typename Q::Block  * W [[buffer(0)]],
                                     device const typename Q::Block  * W_up [[buffer(1)]],
                                     device const typename QS::Block * W_sh [[buffer(2)]],
                                     device const typename QS::Block * W_sh_up [[buffer(3)]],
                                     device const float      * x [[buffer(4)]],
                                     device float            * y [[buffer(5)]],
                                     device const uint4      * tiles [[buffer(6)]],
                                     device const uint       * sorted [[buffer(7)]],
                                     device const float      * route_w [[buffer(8)]],
                                     constant RoutedArgs     & args [[buffer(9)]],
                                     device atomic_uint      * faults [[buffer(10)]],
                                     uint2  tgpig [[threadgroup_position_in_grid]],
                                     ushort tiitg [[thread_index_in_threadgroup]],
                                     ushort sgitg [[simdgroup_index_in_threadgroup]],
                                     ushort tiisg [[thread_index_in_simdgroup]]) {
    threadgroup char shmem[routed_shmem_bytes<T>()];

    if (tgpig.x >= tiles[0].x) {
        return;
    }
    const uint4 tile = tiles[1 + tgpig.x];
    const uint expert = tile.x;
    const uint first = tile.y;
    const short count = (short)tile.z;
    // A tile the table cannot write (a group past the routed experts and the
    // shared group an instance has, a run empty or longer than the tile)
    // raises the expert-id site and stores nothing.
    if (expert >= args.n_experts + BLOOMERY_ROUTER_SHARED || count < 1 || count > ROUTED_ACTS) {
        if (tiitg == 0) {
            raise(faults, args.layer, BLOOMERY_FAULT_SITE_EXPERT_ID);
        }
        return;
    }
    const bool shared = BLOOMERY_ROUTER_SHARED != 0 && expert == args.n_experts;
    if (shared) {
        routed_run<QS, T, PAIR>(shmem, W_sh, PAIR ? W_sh_up : W_sh, tgpig.y, first, count, x, y,
                                sorted, route_w, args, faults, tiitg, sgitg, tiisg);
    } else {
        device const typename Q::Block * wg = W + (ulong)expert * args.stack_blocks;
        device const typename Q::Block * wu = PAIR ? W_up + (ulong)expert * args.stack_blocks : wg;
        routed_run<Q, T, PAIR>(shmem, wg, wu, tgpig.y, first, count, x, y, sorted, route_w, args,
                               faults, tiitg, sgitg, tiisg);
    }
}

#define ROUTED_APART_ENTRY(NAME, Q, BQ, QS, BQS, T, PAIR)                           \
    template [[host_name("gemm_routed_" #NAME)]]                                    \
    kernel void gemm_routed_apart_kernel<Q, QS, T, PAIR>(                           \
        device const BQ  * W [[buffer(0)]],                                         \
        device const BQ  * W_up [[buffer(1)]],                                      \
        device const BQS * W_sh [[buffer(2)]],                                      \
        device const BQS * W_sh_up [[buffer(3)]],                                   \
        device const float    * x [[buffer(4)]],                                    \
        device float          * y [[buffer(5)]],                                    \
        device const uint4    * tiles [[buffer(6)]],                                \
        device const uint     * sorted [[buffer(7)]],                               \
        device const float    * route_w [[buffer(8)]],                              \
        constant RoutedArgs   & args [[buffer(9)]],                                 \
        device atomic_uint    * faults [[buffer(10)]],                               \
        uint2  tgpig [[threadgroup_position_in_grid]],                              \
        ushort tiitg [[thread_index_in_threadgroup]],                               \
        ushort sgitg [[simdgroup_index_in_threadgroup]],                            \
        ushort tiisg [[thread_index_in_simdgroup]]);

#define ROUTED_APART_PAIR(STEM, Q, BQ, QS, BQS)                                                \
    ROUTED_APART_ENTRY(STEM##_pair_f16, Q, BQ, QS, BQS, half, true)                            \
    ROUTED_APART_ENTRY(STEM##_pair_f32, Q, BQ, QS, BQS, float, true)

#define ROUTED_APART_PLAIN(STEM, Q, BQ, QS, BQS)                                               \
    ROUTED_APART_ENTRY(STEM##_plain_f16, Q, BQ, QS, BQS, half, false)                          \
    ROUTED_APART_ENTRY(STEM##_plain_f32, Q, BQ, QS, BQS, float, false)

// The tiles of a shared expert in the stacks' type, per quant.
ROUTED_APART_PAIR(q4k, Q4K, block_q4_K, Q4K, block_q4_K)
ROUTED_APART_PLAIN(q4k, Q4K, block_q4_K, Q4K, block_q4_K)
ROUTED_APART_PAIR(q6k, Q6K, block_q6_K, Q6K, block_q6_K)
ROUTED_APART_PLAIN(q6k, Q6K, block_q6_K, Q6K, block_q6_K)
ROUTED_APART_PAIR(a4, A4, block_mlx_a4, A4, block_mlx_a4)
ROUTED_APART_PLAIN(a4, A4, block_mlx_a4, A4, block_mlx_a4)
ROUTED_APART_PAIR(a8, A8, block_mlx_a8, A8, block_mlx_a8)
ROUTED_APART_PLAIN(a8, A8, block_mlx_a8, A8, block_mlx_a8)
ROUTED_APART_PAIR(a4h, A4H, block_mlx_a4, A4H, block_mlx_a4)
ROUTED_APART_PLAIN(a4h, A4H, block_mlx_a4, A4H, block_mlx_a4)
ROUTED_APART_PAIR(a8h, A8H, block_mlx_a8, A8H, block_mlx_a8)
ROUTED_APART_PLAIN(a8h, A8H, block_mlx_a8, A8H, block_mlx_a8)
ROUTED_APART_PAIR(q8_0, Q8_0, block_q8_0, Q8_0, block_q8_0)
ROUTED_APART_PLAIN(q8_0, Q8_0, block_q8_0, Q8_0, block_q8_0)
ROUTED_APART_PAIR(q5k, Q5K, block_q5_K, Q5K, block_q5_K)
ROUTED_APART_PLAIN(q5k, Q5K, block_q5_K, Q5K, block_q5_K)

// The pairs of types a mixed file has: the pair tile over Q4_K and Q5_K gate
// and up stacks, the plain tile over Q5_K and Q6_K down stacks, each beside a
// Q8_0 shared expert.
ROUTED_APART_PAIR(q4k_apart_q8_0, Q4K, block_q4_K, Q8_0, block_q8_0)
ROUTED_APART_PAIR(q5k_apart_q8_0, Q5K, block_q5_K, Q8_0, block_q8_0)
ROUTED_APART_PLAIN(q5k_apart_q8_0, Q5K, block_q5_K, Q8_0, block_q8_0)
ROUTED_APART_PLAIN(q6k_apart_q8_0, Q6K, block_q6_K, Q8_0, block_q8_0)

// Inkling 2's grouped experts over a pass's route table, the scaled pair and
// plain tiles of its two stack forms:
//
//   gemm_routed_ink2_{nv4p,mx8r,mx8q}_{pair,plain}_f16
//
//   pair     y[p][o] = situ(s_e (gate_e[o] . x), s_e (up_e[o] . x)), the
//            stack's gate rows then its up rows in one tensor (the fused
//            w13 the load interleaved), s_e the expert's global scale
//   plain    y[p][n] = (down_e[n] . x[p]) * (s_e * w[p]), the step's scale
//            of a routed down's finished dot
//
// NV4P is the MLX folder's NVFP4 as two planes an expert (nvfp4.metal's
// planes layout: every row's codes, value 2i the low nibble of byte i, then
// every row's UE4M3 scale bytes, one a 16-value group), MX8R its MXFP8 coda
// rows (33 bytes a 32-value group: the E8M0 scale byte, then 32 E4M3 codes),
// MX8Q the same coda as four-group quads (below). Each form names its row of an expert (`row`) and decodes the 16 values at
// a column of it (`dequant16`), through a `Hold`, the registers a form keeps
// from one K step to the next (empty for the two that read what a step
// needs). Every form decodes exactly into half: an E2M1 code times a UE4M3
// half-scale is at most 7 significant bits inside half's normal range, and
// an E4M3 code times a power of two loses only what falls under half's
// subnormal step (2^-25 of absolute error a weight). The decode is
// nvfp4.metal's rule restated here, since this file comes before it; the
// tile body is the routed tile's, its epilogue the scale and the SITU.
//
// No silent failure: a scale byte that is none decodes NaN; a pair or plain
// output that is not finite raises the product site and is stored NaN.

#pragma METAL fp contract(off)

// The E2M1 significand table, doubled (nvfp4.metal's NV_KVALUES).
static constant float RT_E2M1[16] = {
    0.f, 1.f, 2.f, 3.f, 4.f, 6.f, 8.f, 12.f,
    0.0f, -1.0f, -2.0f, -3.0f, -4.0f, -6.0f, -8.0f, -12.0f,
};

// `ue4m3(b) / 2`, NaN for a byte that is none (nvfp4.metal's nvfp4_scale_half).
inline float rt_ue4m3_half(uint b) {
    if ((b & 0x80u) != 0u || b == 0x7fu) {
        return NAN;
    }
    const uint exp = b >> 3;
    const float man = float(b & 7u);
    const float raw = exp == 0u ? man * exp2(-9.0f) : (1.0f + man / 8.0f) * exp2(float(exp) - 7.0f);
    return 0.5f * raw;
}

// The value of E4M3 code `c`, NaN for the two NaN codes (nvfp4.metal's mxfp8_e4m3).
inline float rt_e4m3(uint c) {
    const uint mag = c & 0x7fu;
    if (mag == 0x7Fu) {
        return NAN;
    }
    const uint exp = mag >> 3;
    const float man = float(mag & 7u);
    const float v = exp == 0u ? man * exp2(-9.0f) : (1.0f + man / 8.0f) * exp2(float(exp) - 7.0f);
    return (c & 0x80u) != 0u ? -v : v;
}

// The SITU activation (situ.metal's pair_situ, this order the gate).
inline float rt_situ(float gate, float up, float gate_beta, float value_beta) {
    const float g = tanh(gate / gate_beta) * gate_beta * (1.0f / (1.0f + exp(-gate)));
    const float u = tanh(up / value_beta) * value_beta;
    return g * u;
}

struct block_mx8r {
    uint8_t e;
    uint8_t q[32];
};

static_assert(sizeof(block_mx8r) == 33, "an MXFP8 row group is 33 bytes");

// A row of an expert's NVFP4 planes: its codes and its scale bytes.
struct nv4p_row {
    device const uint8_t * codes;
    device const uint8_t * scales;
};

struct NV4P {
    typedef nv4p_row Row;
    struct Hold {};

    static inline int values_per_block() {
        return 16;
    }

    // Row `r` of expert `e`: `groups` the 16-value groups of an expert's
    // rows (its codes plane 8 bytes a group, its scales plane after it), `nb`
    // a row's.
    static inline Row row(device const uint8_t * W, ulong groups, uint e, uint nb, uint r) {
        device const uint8_t * base = W + (ulong)e * groups * 9ul;
        return Row{base + (ulong)r * nb * 8ul, base + groups * 8ul + (ulong)r * nb};
    }

    // The 16 values at column `pos` (a multiple of 16) of `row`.
    template <typename T>
    static inline void dequant16(Row row, int pos, thread Hold &, thread matrix<T, 4, 4> & reg) {
        const float s = rt_ue4m3_half(uint(row.scales[pos / 16]));
        const uint2 q = *((device const uint2 *)(row.codes + pos / 2));
        FOR_UNROLL(short i = 0; i < 8; ++i) {
            const uint byte = ((i < 4 ? q.x : q.y) >> (8u * (uint)(i % 4))) & 0xffu;
            reg[(2 * i) / 4][(2 * i) % 4] = (T)(RT_E2M1[byte & 0x0fu] * s);
            reg[(2 * i + 1) / 4][(2 * i + 1) % 4] = (T)(RT_E2M1[byte >> 4u] * s);
        }
    }
};

struct MX8R {
    typedef device const block_mx8r * Row;
    struct Hold {};

    static inline int values_per_block() {
        return 32;
    }

    // Row `r` of expert `e`: `blocks` the groups of an expert's rows, `nb` a
    // row's.
    static inline Row row(device const uint8_t * W, ulong blocks, uint e, uint nb, uint r) {
        return (device const block_mx8r *)W + (ulong)e * blocks + (ulong)r * nb;
    }

    // The 16 values at column `pos` (a multiple of 16) of `row`: half `il`
    // of its group.
    template <typename T>
    static inline void dequant16(Row row, int pos, thread Hold &, thread matrix<T, 4, 4> & reg) {
        device const block_mx8r & blk = row[pos / 32];
        const short il = (short)((pos / 16) % 2);
        const uint sb = uint(blk.e);
        // E8M0: 2^(e - 127), the bias added as its negation (the same exact
        // sum), so the text is the tile's own and not the mat-vec's.
        const float s = sb == 0xffu ? NAN : exp2(float(sb) + (-127.0f));
        FOR_UNROLL(short i = 0; i < 16; ++i) {
            reg[i / 4][i % 4] = (T)(rt_e4m3(uint(blk.q[16 * il + i])) * s);
        }
    }
};

// The value of E4M3 code `c` times the f32 scale `s`, the integer decode of
// nvfp4.metal's mxfp8_quad_value: `mag << 7` is the code's f16 pattern times
// 2^-8 (a subnormal code a subnormal half), so `float(half) * 256` is the
// code's value exactly. The NaN codes (magnitude 0x7f) are NaN, the sign is
// the code's top bit, and the tile then converts the exact f32 product to
// half, as MX8R's does.
inline float rt_mx8q_value(uint c, float s) {
    const uint mag = c & 0x7fu;
    const float e = float(as_type<half>(ushort(mag << 7))) * 256.0f;
    const float m = mag == 0x7fu ? NAN : e;
    return as_type<float>(as_type<uint>(m) ^ ((c & 0x80u) << 24)) * s;
}

// The f32 of E8M0 scale byte `sb`: the byte is its exponent field (byte 0 is
// 2^-127, the f32 pattern 0x00400000; 0xff is NaN), the same value as MX8R's
// `exp2(sb - 127)`.
inline float rt_e8m0(uint sb) {
    return as_type<float>(sb == 0xffu ? 0x7fc00000u : (sb == 0u ? 0x00400000u : (sb << 23)));
}

// MX8Q is the same coda in nvfp4.metal's quads layout: a row is blocks of 132
// bytes, 128 values of four groups — the four groups' E4M3 codes of lane `l`
// the word at byte 4l, then the four scale bytes — so a row's bytes and an
// expert's start are the rows layout's. A thread's 16 values of a K step are
// half `il` of group `gg` of its block: it reads the block's 16 words of its
// half and the four scale bytes once, at the block's first group, and the
// four K steps of the block decode from them (a byte of each word a step),
// where reading a step's words afresh would load four times the bytes the
// step uses. The decode is the integer one (`rt_mx8q_value`).
struct MX8Q {
    typedef device const uint8_t * Row;

    // The block's half a thread holds: its 16 code words and the four scale
    // bytes.
    struct Hold {
        uint w[16];
        uint s;
    };

    static inline int values_per_block() {
        return 128;
    }

    // Row `r` of expert `e`: `blocks` the 128-value blocks of an expert's
    // rows, `nb` a row's.
    static inline Row row(device const uint8_t * W, ulong blocks, uint e, uint nb, uint r) {
        return W + ((ulong)e * blocks + (ulong)r * nb) * 132ul;
    }

    // The 16 values at column `pos` (a multiple of 16) of `row`: half `il`
    // of group `gg` of its block. `hold` carries the block's words from its
    // first group (`gg` 0, where they are read) to its last.
    template <typename T>
    static inline void dequant16(Row row, int pos, thread Hold & hold, thread matrix<T, 4, 4> & reg) {
        const uint gg = (uint)((pos / 32) % 4);
        if (gg == 0u) {
            device const uint8_t * blk = row + (ulong)(pos / 128) * 132ul;
            const uint il = (uint)((pos / 16) % 2);
            device const uint * codes = (device const uint *)blk + 16u * il;
            FOR_UNROLL(short i = 0; i < 16; ++i) {
                hold.w[i] = codes[i];
            }
            hold.s = *(device const uint *)(blk + 128u);
        }
        const float s = rt_e8m0((hold.s >> (8u * gg)) & 0xffu);
        FOR_UNROLL(short i = 0; i < 16; ++i) {
            reg[i / 4][i % 4] = (T)rt_mx8q_value((hold.w[i] >> (8u * gg)) & 0xffu, s);
        }
    }
};

// The scaled tiles' second argument block.
struct Ink2RoutedArgs {
    float gate_beta;
    float value_beta;
    uint per_row;     // slots a pass row holds: the gather's divisor
    uint pad;
};

static_assert(sizeof(Ink2RoutedArgs) == 16, "Ink2RoutedArgs is four words");

// The routed tile (routed_tile's K loop and staging, unchanged) with the
// scaled epilogue: the expert's scale, then SITU on the pair, the scale
// times the slot's route weight on the plain tile.
template <typename Q, typename T, bool PAIR, short NA>
inline void ink2_routed_tile(threadgroup T                  * sa,
                             threadgroup T                  * sb,
                             typename Q::Row                  a_src,
                             threadgroup vec<T, 4>          * a_dst,
                             short                            a_koff,
                             device const float             * b_src,
                             threadgroup vec<T, 4>          * b_dst,
                             short                            count,
                             uint                             first,
                             uint                             r0,
                             device float                   * y,
                             device const uint              * sorted,
                             device const float             * route_w,
                             float                            s_e,
                             constant RoutedArgs            & args,
                             constant Ink2RoutedArgs        & iargs,
                             device atomic_uint             * faults,
                             ushort                           tiitg,
                             ushort                           sgitg,
                             ushort                           tiisg) {
    constexpr int BK = gemm_bk<T>();
    constexpr int LDA = gemm_lda<T>();
    const bool b_stage = (short)(tiitg / 2) < NA * 8;

    const short qid = (short)(tiisg >> 2);
    const short fm = (qid & 4) + ((tiisg >> 1) & 3);
    const short fn = ((qid & 2) << 1) + ((tiisg & 1) << 1);
    const short gn = 16 * (short)sgitg;

    threadgroup const T * arow[NA];
    threadgroup const T * brow[ROUTED_FRAGS_W];
    FOR_UNROLL(short i = 0; i < NA; ++i) {
        arow[i] = sb + fn + (8 * i + fm) * LDA;
    }
    FOR_UNROLL(short j = 0; j < ROUTED_FRAGS_W; ++j) {
        brow[j] = sa + (gn + 8 * j + fn) * LDA + fm;
    }

    vec<float, 2> c[NA][ROUTED_FRAGS_W];
    FOR_UNROLL(short i = 0; i < NA; ++i) {
        FOR_UNROLL(short j = 0; j < ROUTED_FRAGS_W; ++j) {
            c[i][j] = vec<float, 2>(0.f);
        }
    }

    typename Q::Hold hold;
    for (int k0 = 0; k0 < (int)args.cols; k0 += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (b_stage) {
            device const float * xr = b_src + k0;
            gemm_stage_x<T>(xr, b_dst, faults, args.layer);
        }
        {
            matrix<T, 4, 4> reg;
            Q::template dequant16<T>(a_src, k0 + a_koff, hold, reg);
            FOR_UNROLL(short i = 0; i < 4; ++i) {
                a_dst[i] = vec<T, 4>(reg[i][0], reg[i][1], reg[i][2], reg[i][3]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (short kk = 0; kk < BK; kk += 8) {
            simdgroup_barrier(mem_flags::mem_none);
            vec<half, 2> ah[NA];
            vec<half, 2> bh[ROUTED_FRAGS_W];
            FOR_UNROLL(short i = 0; i < NA; ++i) {
                ah[i] = *(threadgroup const vec<half, 2> *)(arow[i] + kk);
            }
            FOR_UNROLL(short j = 0; j < ROUTED_FRAGS_W; ++j) {
                threadgroup const T *w = brow[j];
                bh[j] = vec<half, 2>(w[kk], w[LDA + kk]);
            }
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL(short m = 0; m < NA; ++m) {
                FOR_UNROLL(short n = 0; n < ROUTED_FRAGS_W; ++n) {
                    const short ns = (m % 2) ? ROUTED_FRAGS_W - 1 - n : n;
                    frag_mma_half(c[m][ns], ah[m], bh[ns], c[m][ns]);
                }
            }
        }
    }

    bool bad = false;
    FOR_UNROLL(short i = 0; i < NA; ++i) {
        const short m = 8 * i + fm;
        if (m >= count) {
            continue;
        }
        const uint place = first + (uint)m;
        device float * p = y + (ulong)place * args.y_stride;
        if (PAIR) {
            const uint o = r0 + (uint)(8 * sgitg + fn);
            const float h0 = rt_situ(c[i][0].x * s_e, c[i][1].x * s_e, iargs.gate_beta, iargs.value_beta);
            const float h1 = rt_situ(c[i][0].y * s_e, c[i][1].y * s_e, iargs.gate_beta, iargs.value_beta);
            bad = bad || !isfinite(h0) || !isfinite(h1);
            p[o] = isfinite(h0) ? h0 : NAN;
            p[o + 1] = isfinite(h1) ? h1 : NAN;
        } else {
            const float ws = s_e * route_w[sorted[place]];
            FOR_UNROLL(short j = 0; j < ROUTED_FRAGS_W; ++j) {
                const uint n = r0 + (uint)(gn + 8 * j + fn);
                const float d0 = c[i][j].x * ws;
                const float d1 = c[i][j].y * ws;
                bad = bad || !isfinite(d0) || !isfinite(d1);
                p[n] = isfinite(d0) ? d0 : NAN;
                p[n + 1] = isfinite(d1) ? d1 : NAN;
            }
        }
    }
    if (bad) {
        raise(faults, args.layer, BLOOMERY_FAULT_SITE_F32_PRODUCT);
    }
}

template <typename Q, typename T, bool PAIR>
kernel void gemm_routed_ink2_kernel(device const uint8_t        * W [[buffer(0)]],
                                    device const float      * x [[buffer(1)]],
                                    device float            * y [[buffer(2)]],
                                    device const uint4      * tiles [[buffer(3)]],
                                    device const uint       * sorted [[buffer(4)]],
                                    device const float      * route_w [[buffer(5)]],
                                    constant RoutedArgs     & args [[buffer(6)]],
                                    device atomic_uint      * faults [[buffer(7)]],
                                    device const float      * scales [[buffer(8)]],
                                    constant Ink2RoutedArgs & iargs [[buffer(9)]],
                                    uint2  tgpig [[threadgroup_position_in_grid]],
                                    ushort tiitg [[thread_index_in_threadgroup]],
                                    ushort sgitg [[simdgroup_index_in_threadgroup]],
                                    ushort tiisg [[thread_index_in_simdgroup]]) {
    threadgroup char shmem[routed_shmem_bytes<T>()];
    constexpr int LDA = gemm_lda<T>();

    if (tgpig.x >= tiles[0].x) {
        return;
    }
    const uint4 tile = tiles[1 + tgpig.x];
    const uint expert = tile.x;
    const uint first = tile.y;
    const short count = (short)tile.z;
    if (expert >= args.n_experts || count < 1 || count > ROUTED_ACTS) {
        if (tiitg == 0) {
            raise(faults, args.layer, (BLOOMERY_FAULT_SITE_EXPERT_ID));
        }
        return;
    }
    const float s_e = scales[expert];
    const uint r0 = tgpig.y * (PAIR ? ROUTED_ROWS / 2 : ROUTED_ROWS);
    const uint nb = args.cols / (uint)Q::values_per_block();

    threadgroup T * sa = (threadgroup T *)shmem;
    threadgroup T * sb = sa + ROUTED_ROWS * LDA;

    const short a_row = (short)(tiitg / 2);
    const short a_koff = (short)((tiitg % 2) * 16);
    typename Q::Row a_src;
    if (PAIR) {
        // The fused stack's up rows follow its gate rows.
        const uint o = r0 + (uint)(8 * (a_row / 16) + a_row % 8);
        a_src = Q::row(W, args.stack_blocks, expert, nb, ((a_row / 8) % 2 == 0 ? 0u : args.rows) + o);
    } else {
        a_src = Q::row(W, args.stack_blocks, expert, nb, r0 + (uint)a_row);
    }
    threadgroup vec<T, 4> * a_dst = (threadgroup vec<T, 4> *)(sa + a_row * LDA + a_koff);

    const short b_row = (short)(tiitg / 2);
    const uint b_place = first + (uint)min(b_row, (short)(count - 1));
    const ulong src_row =
        args.gather != 0u ? (ulong)(sorted[b_place] / iargs.per_row) : (ulong)b_place;
    device const float * b_src = x + src_row * args.x_stride + a_koff;
    threadgroup vec<T, 4> * b_dst = (threadgroup vec<T, 4> *)(sb + b_row * LDA + a_koff);

    switch ((count + 7) >> 3) {
#define INK2_ROUTED_CASE(NA)                                                              \
    case NA:                                                                              \
        ink2_routed_tile<Q, T, PAIR, NA>(sa, sb, a_src, a_dst, a_koff, b_src, b_dst,      \
                                         count, first, r0, y, sorted, route_w, s_e, args, \
                                         iargs, faults, tiitg, sgitg, tiisg);             \
        break;
        INK2_ROUTED_CASE(1)
        INK2_ROUTED_CASE(2)
        INK2_ROUTED_CASE(3)
        INK2_ROUTED_CASE(4)
        INK2_ROUTED_CASE(5)
        INK2_ROUTED_CASE(6)
        INK2_ROUTED_CASE(7)
        INK2_ROUTED_CASE(8)
#undef INK2_ROUTED_CASE
    }
}

#define INK2_ROUTED_ENTRY(NAME, Q, PAIR)                                            \
    template [[host_name("gemm_routed_ink2_" #NAME)]]                               \
    kernel void gemm_routed_ink2_kernel<Q, half, PAIR>(                             \
        device const uint8_t    * W [[buffer(0)]],                                  \
        device const float      * x [[buffer(1)]],                                  \
        device float            * y [[buffer(2)]],                                  \
        device const uint4      * tiles [[buffer(3)]],                              \
        device const uint       * sorted [[buffer(4)]],                             \
        device const float      * route_w [[buffer(5)]],                            \
        constant RoutedArgs     & args [[buffer(6)]],                               \
        device atomic_uint      * faults [[buffer(7)]],                             \
        device const float      * scales [[buffer(8)]],                             \
        constant Ink2RoutedArgs & iargs [[buffer(9)]],                              \
        uint2  tgpig [[threadgroup_position_in_grid]],                              \
        ushort tiitg [[thread_index_in_threadgroup]],                               \
        ushort sgitg [[simdgroup_index_in_threadgroup]],                            \
        ushort tiisg [[thread_index_in_simdgroup]]);

INK2_ROUTED_ENTRY(nv4p_pair_f16, NV4P, true)
INK2_ROUTED_ENTRY(nv4p_plain_f16, NV4P, false)
INK2_ROUTED_ENTRY(mx8r_pair_f16, MX8R, true)
INK2_ROUTED_ENTRY(mx8r_plain_f16, MX8R, false)
INK2_ROUTED_ENTRY(mx8q_pair_f16, MX8Q, true)
INK2_ROUTED_ENTRY(mx8q_plain_f16, MX8Q, false)
