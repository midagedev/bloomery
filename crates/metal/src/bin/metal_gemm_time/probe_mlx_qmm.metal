// Probe-only: MLX's affine `qmm_t` for 4-bit, group-64 weights, the
// reference arm of metal_gemm_time (`op=mlx16`, `op=mlx32`). This text is
// compiled by the probe into a library of its own and never enters the
// engine's kernel source.
//
// Source: ml-explore/mlx v0.32.1, MIT licence, Copyright (c) 2023-2024
// Apple Inc. The code is theirs, reduced to the one instantiation the probe
// launches (the transposed weights, 4 bits, group 64, a full tile in both
// directions, one batch) and with the template machinery that instantiation
// does not read taken out:
//   mlx/backend/metal/kernels/quantized.h          qmm_t_impl,
//                                                  QuantizedBlockLoader,
//                                                  dequantize (the 4-bit arm)
//   mlx/backend/metal/kernels/steel/gemm/mma.h     BaseMMAFrag, MMATile,
//                                                  tile_matmad, BlockMMA
//   mlx/backend/metal/kernels/steel/gemm/loader.h  BlockLoader
// Launch (mlx/backend/metal/quantized.cpp `qmm`): BM = BN = BK = 32,
// WM = WN = 2, a threadgroup of 128 threads, the grid N/32 tiles across by
// M/32 down. The layout is MLX's: the codes as u32 words, row-major, value
// 8w + j the nibble j of word w; the scales and biases as separate arrays,
// one pair per 64 values of a row; x and y in the operand type.
//
// The licence notice of the upstream files is carried by the lead into
// THIRD_PARTY_NOTICES.md when this file lands.

#include <metal_stdlib>
#include <metal_simdgroup_matrix>

using namespace metal;

#define STEEL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")
#define MLXREF_FUNC inline __attribute__((__always_inline__))

#define MLXREF_BM 32
#define MLXREF_BN 32
#define MLXREF_BK 32
#define MLXREF_WM 2
#define MLXREF_WN 2
#define MLXREF_THREADS 128
#define MLXREF_GROUP 64

struct MlxrefArgs {
    int K;
    int N;
    int M;
};

// BaseMMAFrag<T, 8, 8>: the lane's two elements of an 8x8 fragment.
template <typename T>
struct RefFrag {
    typedef simdgroup_matrix<T, 8, 8> mat_type;
    typedef vec<T, 2> frag_type;

    MLXREF_FUNC static short2 get_coord(ushort simd_lane_id) {
        const short qid = simd_lane_id / 4;
        const short fm = (qid & 4) + ((simd_lane_id / 2) % 4);
        const short fn = (qid & 2) * 2 + (simd_lane_id % 2) * 2;
        return short2{fn, fm};
    }

    template <typename SrcPtr>
    MLXREF_FUNC static void load(thread frag_type & dst, SrcPtr src, short str_y) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < 2; j++) {
            dst[j] = static_cast<T>(src[j * str_y]);
        }
    }

    MLXREF_FUNC static void mma(thread frag_type & D,
                                thread frag_type & A,
                                thread frag_type & B,
                                thread frag_type & C) {
        mat_type D_mat;
        mat_type A_mat;
        mat_type B_mat;
        mat_type C_mat;
        reinterpret_cast<thread frag_type &>(A_mat.thread_elements()) = A;
        reinterpret_cast<thread frag_type &>(B_mat.thread_elements()) = B;
        reinterpret_cast<thread frag_type &>(C_mat.thread_elements()) = C;
        simdgroup_multiply_accumulate(D_mat, A_mat, B_mat, C_mat);
        D = reinterpret_cast<thread frag_type &>(D_mat.thread_elements());
    }
};

// BlockMMA<T, T, 32, 32, 32, 2, 2, false, true, BK_padded, BK_padded>: the
// fragments are the float accumulator type, loaded (converted) from the
// threadgroup tiles of the operand type T.
template <typename T>
struct RefBlockMMA {
    typedef RefFrag<float> Frag;
    typedef vec<float, 2> frag_type;

    static constant constexpr short TM = MLXREF_BM / (8 * MLXREF_WM);
    static constant constexpr short TN = MLXREF_BN / (8 * MLXREF_WN);
    static constant constexpr short LD = MLXREF_BK + 16 / (short)sizeof(T);

    frag_type Atile[TM];
    frag_type Btile[TN];
    frag_type Ctile[TM * TN];

    short sm;
    short sn;
    short As_offset;
    short Bs_offset;

    MLXREF_FUNC RefBlockMMA(ushort simd_group_id, ushort simd_lane_id) {
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < TM * TN; ++i) {
            Ctile[i] = frag_type(0);
        }
        const short tm = 8 * (simd_group_id / MLXREF_WN);
        const short tn = 8 * (simd_group_id % MLXREF_WN);
        const short2 simd_coord = Frag::get_coord(simd_lane_id);
        sm = simd_coord.y;
        sn = simd_coord.x;
        As_offset = (tm + sm) * LD + sn;
        Bs_offset = sm + (tn + sn) * LD;
        sm += tm;
        sn += tn;
    }

    MLXREF_FUNC void mma(const threadgroup T * As, const threadgroup T * Bs) {
        As += As_offset;
        Bs += Bs_offset;
        STEEL_PRAGMA_UNROLL
        for (short kk = 0; kk < MLXREF_BK; kk += 8) {
            simdgroup_barrier(mem_flags::mem_none);
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < TM; ++i) {
                Frag::load(Atile[i], &As[(i * 8) * MLXREF_WM * LD], 1);
            }
            simdgroup_barrier(mem_flags::mem_none);
            STEEL_PRAGMA_UNROLL
            for (short j = 0; j < TN; ++j) {
                Frag::load(Btile[j], &Bs[(j * 8) * MLXREF_WN * LD], LD);
            }
            simdgroup_barrier(mem_flags::mem_none);
            STEEL_PRAGMA_UNROLL
            for (short m = 0; m < TM; ++m) {
                STEEL_PRAGMA_UNROLL
                for (short n = 0; n < TN; ++n) {
                    const short n_serp = (m % 2) ? (TN - 1 - n) : n;
                    Frag::mma(Ctile[m * TN + n_serp], Atile[m], Btile[n_serp],
                              Ctile[m * TN + n_serp]);
                }
            }
            As += 8;
            Bs += 8;
        }
    }

    template <typename U>
    MLXREF_FUNC void store_result(device U * D, const int ldd) {
        D += sm * ldd + sn;
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < TM; ++i) {
            STEEL_PRAGMA_UNROLL
            for (short j = 0; j < TN; ++j) {
                device U * p = &D[(i * 8) * MLXREF_WM * ldd + (j * 8) * MLXREF_WN];
                STEEL_PRAGMA_UNROLL
                for (short e = 0; e < 2; ++e) {
                    p[e] = static_cast<U>(Ctile[i * TN + j][e]);
                }
            }
        }
    }
};

// BlockLoader<T, 32, 32, BK_padded, 1, 128>: x, 8 values a thread, one
// vector a row piece.
template <typename T>
struct RefXLoader {
    static constant constexpr short LD = MLXREF_BK + 16 / (short)sizeof(T);
    static constant constexpr short n_reads = (MLXREF_BK * MLXREF_BM) / MLXREF_THREADS;
    static constant constexpr short TCOLS = MLXREF_BK / n_reads;
    static constant constexpr short TROWS = MLXREF_THREADS / TCOLS;

    struct alignas(sizeof(T)) ReadVector {
        uint8_t v[sizeof(T) * n_reads];
    };

    const int src_ld;
    const short bi;
    const short bj;
    threadgroup T * dst;
    const device T * src;

    MLXREF_FUNC RefXLoader(const device T * src_, const int src_ld_, threadgroup T * dst_,
                           ushort simd_group_id, ushort simd_lane_id)
        : src_ld(src_ld_),
          bi((simd_group_id * 32 + simd_lane_id) / TCOLS),
          bj(n_reads * ((simd_group_id * 32 + simd_lane_id) % TCOLS)),
          dst(dst_ + bi * LD + bj),
          src(src_ + bi * src_ld_ + bj) {}

    MLXREF_FUNC void load_unsafe() const {
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < MLXREF_BM; i += TROWS) {
            *((threadgroup ReadVector *)(&dst[i * LD])) = *((const device ReadVector *)(&src[i * src_ld]));
        }
    }

    MLXREF_FUNC void next() { src += MLXREF_BK; }
};

// QuantizedBlockLoader<T, 32, 32, BK_padded, 1, 128, 64, 4>: the weights,
// four packed bytes a thread, dequantized into the staged tile.
template <typename T>
struct RefWLoader {
    static constant constexpr short LD = MLXREF_BK + 16 / (short)sizeof(T);
    static constant constexpr short pack_factor = 2;
    static constant constexpr short BCOLS_PACKED = MLXREF_BK / pack_factor;
    static constant constexpr short n_reads = (BCOLS_PACKED * MLXREF_BN) / MLXREF_THREADS;
    static constant constexpr short group_steps = MLXREF_GROUP / MLXREF_BK;

    short group_step_cnt;
    const short bi;
    const short bj;
    threadgroup T * dst;
    const device uint8_t * src;
    const device T * scales;
    const device T * biases;

    MLXREF_FUNC RefWLoader(const device uint8_t * src_, const device T * scales_,
                           const device T * biases_, const int src_ld_, threadgroup T * dst_,
                           ushort simd_group_id, ushort simd_lane_id)
        : group_step_cnt(0),
          bi(n_reads * (simd_group_id * 32 + simd_lane_id) / BCOLS_PACKED),
          bj((n_reads * (simd_group_id * 32 + simd_lane_id)) % BCOLS_PACKED),
          dst(dst_ + bi * LD + bj * pack_factor),
          src(src_ + bi * src_ld_ / pack_factor + bj),
          scales(scales_ + bi * src_ld_ / MLXREF_GROUP),
          biases(biases_ + bi * src_ld_ / MLXREF_GROUP) {}

    MLXREF_FUNC void load_unsafe() const {
        const T scale = *scales;
        const T bias = *biases;
        const float s = float(scale);
        const float b = float(bias);
        const float sc[2] = {s, s / 16.0f};
        for (int i = 0; i < n_reads; i++) {
            const device uint8_t * w = src + i;
            threadgroup T * w_local = dst + i * pack_factor;
            w_local[0] = static_cast<T>(sc[0] * (w[0] & 0x0f) + b);
            w_local[1] = static_cast<T>(sc[1] * (w[0] & 0xf0) + b);
        }
    }

    MLXREF_FUNC void next() {
        src += BCOLS_PACKED;
        if (group_steps > 1) {
            group_step_cnt++;
            if (group_step_cnt == group_steps) {
                group_step_cnt = 0;
                scales++;
                biases++;
            }
        } else {
            scales++;
            biases++;
        }
    }
};

template <typename T>
kernel void mlxref_qmm_t(const device uint32_t * w [[buffer(0)]],
                         const device T * scales [[buffer(1)]],
                         const device T * biases [[buffer(2)]],
                         const device T * x [[buffer(3)]],
                         device T * y [[buffer(4)]],
                         constant MlxrefArgs & args [[buffer(5)]],
                         uint3 tid [[threadgroup_position_in_grid]],
                         ushort simd_gid [[simdgroup_index_in_threadgroup]],
                         ushort simd_lid [[thread_index_in_simdgroup]]) {
    constexpr short LD = MLXREF_BK + 16 / (short)sizeof(T);
    threadgroup T Xs[MLXREF_BM * LD];
    threadgroup T Ws[MLXREF_BN * LD];

    const int K = args.K;
    const int N = args.N;
    const int K_w = K / 2;
    const int K_g = K / MLXREF_GROUP;
    const int y_row = tid.y * MLXREF_BM;
    const int y_col = tid.x * MLXREF_BN;

    auto wl = (const device uint8_t *)w;
    x += y_row * static_cast<int64_t>(K);
    wl += y_col * static_cast<int64_t>(K_w);
    scales += y_col * static_cast<int64_t>(K_g);
    biases += y_col * static_cast<int64_t>(K_g);
    y += y_row * static_cast<int64_t>(N) + y_col;

    RefXLoader<T> loader_x(x, K, Xs, simd_gid, simd_lid);
    RefWLoader<T> loader_w(wl, scales, biases, K, Ws, simd_gid, simd_lid);
    RefBlockMMA<T> mma_op(simd_gid, simd_lid);

    for (int k = 0; k < K; k += MLXREF_BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_unsafe();
        loader_w.load_unsafe();
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    mma_op.store_result(y, N);
}

template [[host_name("mlxref_qmm_t_f16")]] [[kernel]] decltype(mlxref_qmm_t<half>) mlxref_qmm_t<half>;
template [[host_name("mlxref_qmm_t_f32")]] [[kernel]] decltype(mlxref_qmm_t<float>) mlxref_qmm_t<float>;
