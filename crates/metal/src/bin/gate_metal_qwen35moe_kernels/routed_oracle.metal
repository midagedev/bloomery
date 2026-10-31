// The routed GEMM's oracle (clause grouped_oracle): the grouped GEMM's tile
// at a 32-slot table tile, each SIMD group multiplying two 16-row fragments
// of activation rows (8 rows each, a test a fragment for the run's length)
// against two 16-row fragments of weight rows. The gate compiles it in place
// of gemm_routed.metal's routed tile, from `#define ROUTED_FRAGS_A` on, with
// the table's tile and gemm_shared.metal's ROUTED_ACTS at 32 (the tile's
// weight rows and its argument block are gemm_shared.metal's), and holds the
// shipped tile to its outputs bit for bit: a tile's height and its split over
// the SIMD groups leave every output's K order as it is.

#define ROUTED_FRAGS_A (ROUTED_ACTS / 16)
#define ROUTED_FRAGS_W (ROUTED_ROWS / 16)

static_assert(ROUTED_ACTS == ROUTE_TILE, "a tile holds a table tile's slots");
static_assert(ROUTED_ROWS * 2 == GEMM_THREADS,
              "a thread stages one 16-value chunk of a weight row");

template <typename T>
constexpr int routed_shmem_bytes() {
    return (ROUTED_ROWS + ROUTED_ACTS) * gemm_lda<T>() * (int)sizeof(T);
}

// A tile over quant Q from the group's gate (and up) tensors: the body the
// kernel below takes, the shipped kernel's prologue left to it.
template <typename Q, typename T, bool PAIR>
inline void oracle_run(threadgroup char                * shmem,
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
    constexpr int BK = gemm_bk<T>();
    constexpr int LDA = gemm_lda<T>();

    // The tile's first output: the pair's tile is 32 outputs (64 weight rows
    // of two tensors), the plain tile 64 rows of one.
    const uint r0 = tile_y * (PAIR ? ROUTED_ROWS / 2 : ROUTED_ROWS);

    const uint nb = args.cols / (uint)Q::values_per_block();

    threadgroup T * sa = (threadgroup T *)shmem;
    threadgroup T * sb = sa + ROUTED_ROWS * LDA;

    // A: a thread stages one 16-value chunk of a weight row a K step.
    const short a_row = (short)(tiitg / 2);
    const short a_koff = (short)((tiitg % 2) * 16);
    device const typename Q::Block * a_src;
    if (PAIR) {
        a_src = (a_row < ROUTED_ROWS / 2 ? wg : wu) + (ulong)(r0 + (a_row % (ROUTED_ROWS / 2))) * nb;
    } else {
        a_src = wg + (ulong)(r0 + a_row) * nb;
    }
    threadgroup vec<T, 4> * a_dst = (threadgroup vec<T, 4> *)(sa + a_row * LDA + a_koff);

    // B: the first 64 threads stage one 16-value chunk of an activation row;
    // a row past the run's length takes the run's last (a defined access).
    const bool b_live = tiitg < ROUTED_ACTS * 2;
    const short b_row = (short)(tiitg / 2);
    device const float * b_src = x;
    threadgroup vec<T, 4> * b_dst = (threadgroup vec<T, 4> *)(sb);
    if (b_live) {
        const uint place = first + (uint)min(b_row, (short)(count - 1));
        const ulong src_row = args.gather != 0u ? (ulong)(sorted[place] / BLOOMERY_EXPERT_SLOTS) : (ulong)place;
        b_src = x + src_row * args.x_stride + a_koff;
        b_dst = (threadgroup vec<T, 4> *)(sb + b_row * LDA + a_koff);
    }

    const short qid = (short)(tiisg >> 2);
    const short fm = (qid & 4) + ((tiisg >> 1) & 3);
    const short fn = ((qid & 2) << 1) + ((tiisg & 1) << 1);
    const short gm = 8 * (sgitg >> 1);
    const short gn = 8 * (sgitg & 1);

    // A fragment row past the run's length is not loaded or multiplied.
    bool live[ROUTED_FRAGS_A];
    threadgroup const T * arow[ROUTED_FRAGS_A];
    threadgroup const T * brow[ROUTED_FRAGS_W];
    FOR_UNROLL(short i = 0; i < ROUTED_FRAGS_A; ++i) {
        live[i] = gm + 16 * i < count;
        arow[i] = sb + (gm + 16 * i + fm) * LDA + fn;
    }
    FOR_UNROLL(short j = 0; j < ROUTED_FRAGS_W; ++j) {
        brow[j] = sa + (gn + 16 * j + fn) * LDA + fm;
    }

    vec<float, 2> c[ROUTED_FRAGS_A][ROUTED_FRAGS_W];
    FOR_UNROLL(short i = 0; i < ROUTED_FRAGS_A; ++i) {
        FOR_UNROLL(short j = 0; j < ROUTED_FRAGS_W; ++j) {
            c[i][j] = vec<float, 2>(0.f);
        }
    }

    for (int k0 = 0; k0 < (int)args.cols; k0 += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (b_live) {
            device const float * xr = b_src + k0;
            FOR_UNROLL(short g = 0; g < 4; ++g) {
                const float4 v = float4(xr[4 * g], xr[4 * g + 1], xr[4 * g + 2], xr[4 * g + 3]);
                if (sizeof(T) == 2) {
                    const bool4 over = !(fabs(v) <= 65504.f);
                    if (any(over)) {
                        raise(faults, args.layer, BLOOMERY_FAULT_SITE_QUANT_COLUMN);
                        b_dst[g] = vec<T, 4>(select(v, float4(NAN), over));
                    } else {
                        b_dst[g] = vec<T, 4>(v);
                    }
                } else {
                    b_dst[g] = vec<T, 4>(v);
                }
            }
        }
        {
            const int pos = k0 + a_koff;
            const int ib = pos / Q::values_per_block();
            const short il = (short)((pos / 16) % (Q::values_per_block() / 16));
            matrix<T, 4, 4> reg;
            Q::template dequant16<T>(a_src[ib], il, reg);
            FOR_UNROLL(short i = 0; i < 4; ++i) {
                a_dst[i] = vec<T, 4>(reg[i][0], reg[i][1], reg[i][2], reg[i][3]);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (short kk = 0; kk < BK; kk += 8) {
            simdgroup_barrier(mem_flags::mem_none);
            vec<float, 2> af[ROUTED_FRAGS_A];
            vec<float, 2> bf[ROUTED_FRAGS_W];
            vec<half, 2> ah[ROUTED_FRAGS_A];
            vec<half, 2> bh[ROUTED_FRAGS_W];
            if (sizeof(T) == 2) {
                FOR_UNROLL(short i = 0; i < ROUTED_FRAGS_A; ++i) {
                    if (live[i]) {
                        ah[i] = *(threadgroup const vec<half, 2> *)(arow[i] + kk);
                    }
                }
                FOR_UNROLL(short j = 0; j < ROUTED_FRAGS_W; ++j) {
                    threadgroup const T *w = brow[j];
                    bh[j] = vec<half, 2>(w[kk], w[LDA + kk]);
                }
            } else {
                FOR_UNROLL(short i = 0; i < ROUTED_FRAGS_A; ++i) {
                    if (live[i]) {
                        af[i] = *(threadgroup const vec<float, 2> *)(arow[i] + kk);
                    }
                }
                FOR_UNROLL(short j = 0; j < ROUTED_FRAGS_W; ++j) {
                    threadgroup const T *w = brow[j];
                    bf[j] = vec<float, 2>(w[kk], w[LDA + kk]);
                }
            }
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL(short m = 0; m < ROUTED_FRAGS_A; ++m) {
                if (live[m]) {
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
        }
    }

    // The store, per element under the run's length. A pair's lane holds an
    // output's gate in fragment column j (j < 2) and its up in j + 2.
    FOR_UNROLL(short i = 0; i < ROUTED_FRAGS_A; ++i) {
        const short m = gm + 16 * i + fm;
        if (m >= count) {
            continue;
        }
        const uint place = first + (uint)m;
        device float * p = y + (ulong)place * args.y_stride;
        if (PAIR) {
            const float rw = route_w[sorted[place]];
            FOR_UNROLL(short j = 0; j < ROUTED_FRAGS_W / 2; ++j) {
                const uint o = r0 + (uint)(gn + 16 * j + fn);
                p[o] = pair_swiglu(c[i][j].x, c[i][j + 2].x) * rw;
                p[o + 1] = pair_swiglu(c[i][j].y, c[i][j + 2].y) * rw;
            }
        } else {
            FOR_UNROLL(short j = 0; j < ROUTED_FRAGS_W; ++j) {
                const uint n = r0 + (uint)(gn + 16 * j + fn);
                p[n] = c[i][j].x;
                p[n + 1] = c[i][j].y;
            }
        }
    }
}

// The shipped kernel's shape over the oracle's tile: the library the host
// builds its kernels from holds every entry the shipped one does.
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
    if (expert >= args.n_experts + BLOOMERY_ROUTER_SHARED) {
        if (tiitg == 0) {
            raise(faults, args.layer, BLOOMERY_FAULT_SITE_EXPERT_ID);
        }
        return;
    }
    if (BLOOMERY_ROUTER_SHARED != 0 && expert == args.n_experts) {
        oracle_run<QS, T, PAIR>(shmem, W_sh, PAIR ? W_sh_up : W_sh, tgpig.y, first, count, x, y,
                                sorted, route_w, args, faults, tiitg, sgitg, tiisg);
    } else {
        device const typename Q::Block * wg = W + (ulong)expert * args.stack_blocks;
        device const typename Q::Block * wu = PAIR ? W_up + (ulong)expert * args.stack_blocks : wg;
        oracle_run<Q, T, PAIR>(shmem, wg, wu, tgpig.y, first, count, x, y, sorted, route_w, args,
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
ROUTED_APART_PAIR(q4k_apart_q8_0, Q4K, block_q4_K, Q8_0, block_q8_0)
ROUTED_APART_PAIR(q5k_apart_q8_0, Q5K, block_q5_K, Q8_0, block_q8_0)
ROUTED_APART_PLAIN(q5k_apart_q8_0, Q5K, block_q5_K, Q8_0, block_q8_0)
ROUTED_APART_PLAIN(q6k_apart_q8_0, Q6K, block_q6_K, Q8_0, block_q8_0)
