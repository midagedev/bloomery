// The grouped GEMM of the routed experts, and the dense GEMM over f32 weight
// rows. The entry points, under their host names:
//
//   gemm_routed_{q4k,q6k,a4,a8,q8_0,q5k}_{pair,plain}_{f16,f32}
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
