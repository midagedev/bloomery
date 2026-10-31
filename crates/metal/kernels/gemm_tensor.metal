// The prompt pass's GEMMs on Metal 4's tensor ops (MetalPerformancePrimitives'
// `matmul2d`), the path a GPU whose cores carry matrix units takes. The
// entry points, under the host names of the simdgroup kernels they stand in
// for (one library or the other is the GEMMs' library; `Library` picks):
//
//   gemm_{q4k,q6k,a4,a8,a4h,a8h,q8_0,q5k}_{store,resid}[_seg]_{f16,f32}     gemm.metal's
//   gemm_routed_{q4k,q6k,a4,a8,a4h,a8h,q8_0,q5k}_{pair,plain}_{f16,f32}     gemm_routed.metal's
//
// Same arguments, buffers, grid and meanings as those (their headers hold
// them): the epilogues, the segment table, the route table's tiles, the f16
// operand's checked cast and its fault, the expert-id fault. What differs is
// the multiply and its sum order.
//
// The design is MLX's NAX quantized matmul (MIT licence; see
// THIRD_PARTY_NOTICES.md), the path MLX takes where `is_nax_available`:
//   mlx/backend/metal/kernels/quantized_nax.h:935-1078  qmm_t_nax_tgp_impl:
//       the weights dequantized into threadgroup memory a K step, the 64x64
//       tile split over a 2x2 grid of SIMD groups (WM = WN = 2), each SIMD
//       group's 32x32 block of the output its own op's
//   mlx/backend/metal/kernels/steel/gemm/nax.h:387-411  the descriptor:
//       matmul2d_descriptor(m, n, k, transpose_left = false,
//       transpose_right = true, ..., mode::multiply_accumulate) at
//       execution_simdgroup scope, the weight operand [n][k] (transposed)
// What differs from MLX:
//   - the activations are staged too, in the simdgroup kernels' layout (the
//     same K step of 32 and pitch): an f32 activation is cast to half with
//     the range check that raises the activation quantizer's site, which
//     needs the value in hand; MLX loads its already-half activations from
//     device memory straight into registers. MLX's K step is 64 with only
//     the weights staged (9 KB); ours is 32 with both (10 KB for f16), the
//     same threadgroup bytes, so as many tiles sit on a core;
//   - the op's operands are tensors over the staged rows and its
//     destination a cooperative tensor read back through
//     `get_multidimensional_index`, the layout the API defines; MLX loads
//     its fragments into the cooperative tensors' storage in the layout its
//     NAX fragment assumes (`BaseNAXFrag::get_coord`), which only the
//     hardware it targets defines. A dense SIMD group's op is the whole
//     32x32x32 step, not MLX's 16x32x16 pieces; a routed group's ops are
//     16 places high, MLX's fragment height, so a short run wastes less;
//   - relaxed precision is off (MLX's descriptor sets it): the clauses'
//     band is derived for the multiply at its defined precision.
// Where llama.cpp's tensor `kernel_mul_mm` (ggml-metal/kernels/mul_mm.metal,
// under GGML_METAL_HAS_TENSOR) differs: one op over the whole threadgroup
// (execution_simdgroups<4>, a 64-weight by 128-activation tile, K dynamic),
// the f32 activations read from device memory as the op's left operand with
// no cast, relaxed precision on, and the destination stored through
// `cooperative_tensor::store`.
//
// Geometry: GEMM_THREADS (128) threads in four SIMD groups; a tile is
// GEMM_ROWS (64) weight rows by GEMM_ACTS (64) activation rows, a K step
// gemm_bk (32) values, both staged [row][k] at gemm_lda's pitch (the
// weights at the top, the activations under them). SIMD group g owns the
// activation rows 32 (g >> 1) on by the weight rows 32 (g & 1) on: one
// matmul2d a K step, accumulating into its cooperative tensor. A thread
// stages one 16-value chunk of each side a K step (one `dequant16`, one
// checked float4 run); rows past the tile's last live row stage that row's
// values (a defined access) and are not stored. The store is per element,
// under each element's own row and column bound. A SIMD group whose
// activation rows are all past the tile's live rows runs no op and stores
// nothing.
//
// The routed tile (gemm_routed.metal's): ROUTED_ROWS (64) weight rows by
// ROUTED_ACTS (64) slot places, the table's tile a threadgroup. SIMD group g
// owns the places 32 (g >> 1) on, as two blocks of TENSOR_RM (16) places
// (MLX's fragment height), each its own op, run only when the run reaches
// the block: a run's places are multiplied in 16-place steps, so a short
// run (the routed tiles' common case) wastes at most 15 places, and only
// the reached blocks' activation rows are staged. The pair's weight rows
// are the gate of 32 outputs, then their up; group g multiplies its places
// against 16 outputs' gate and the same 16 outputs' up, two ops of one
// type, so the two destinations share one layout and element i of each is
// one output's gate and up. The plain tile's group multiplies its places by
// 32 down rows.
//
// The sum order of an output is the op's: the clauses hold these kernels to
// a band derived for any order of its K products, not to the simdgroup
// kernels' bits.
//
// This file comes after quant.metal (the quant owners), fault.metal (raise)
// and gemm_shared.metal (the geometry, the argument blocks, pair_swiglu,
// the activations' checked cast gemm_stage_x).

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

#pragma METAL fp contract(off)

// The tensor op's precision: relaxed precision lets the implementation trade
// accuracy for speed by its own rule, which no band is derived for.
#define TENSOR_RELAXED false

// A SIMD group's block of the dense tile: half of each side (MLX's WM = WN = 2).
#define TENSOR_SM (GEMM_ACTS / 2)
#define TENSOR_SN (GEMM_ROWS / 2)
// A pair group's outputs: half of the tile's 32 outputs.
#define TENSOR_PAIR_N (ROUTED_ROWS / 4)
// A routed op's places: half of a SIMD group's 32.
#define TENSOR_RM (TENSOR_SM / 2)

static_assert(GEMM_THREADS == 4 * 32, "four SIMD groups of 32 threads");
static_assert(GEMM_ROWS * 2 == GEMM_THREADS && GEMM_ACTS * 2 == GEMM_THREADS,
              "a thread stages one 16-value chunk of a weight row and of an activation row");
static_assert(ROUTED_ROWS == GEMM_ROWS && ROUTED_ACTS == GEMM_ACTS,
              "the routed tile is the dense tile's shape");
static_assert(TENSOR_RM * 2 == TENSOR_SM, "a routed SIMD group's places are two ops' blocks");
static_assert(gemm_bk<half>() == 32 && gemm_bk<float>() == 32,
              "a K step is two 16-value chunks, the op's static K");

// The op of one SIMD group: an m-by-n block of the output over a K step,
// the left operand (the activations) [m][k], the right (the weights) [n][k].
template <int M, int N>
using tensor_mm = matmul2d<matmul2d_descriptor(M, N, 32, false, true, TENSOR_RELAXED,
                                               matmul2d_descriptor::mode::multiply_accumulate),
                           execution_simdgroup>;

// `rows` staged rows of one K step at `at`, as the op's operand: K the
// contiguous extent, the rows at the staged pitch.
template <typename T, int ROWS>
inline auto tensor_rows(threadgroup T * at) {
    return tensor(at, extents<int32_t, 32, ROWS>(), array<int32_t, 2>({1, gemm_lda<T>()}));
}

// One 16-value chunk of a weight row: one `dequant16` of the quant owner,
// staged as four contiguous vectors.
template <typename Q, typename T>
inline void tensor_stage_w(device const typename Q::Block * row, int pos,
                           threadgroup vec<T, 4> * dst) {
    const int ib = pos / Q::values_per_block();
    const short il = (short)((pos / 16) % (Q::values_per_block() / 16));
    matrix<T, 4, 4> reg;
    Q::template dequant16<T>(row[ib], il, reg);
    FOR_UNROLL(short i = 0; i < 4; ++i) {
        dst[i] = vec<T, 4>(reg[i][0], reg[i][1], reg[i][2], reg[i][3]);
    }
}

// The cooperative tensor's elements a lane holds, zeroed.
template <typename C>
inline void tensor_zero(thread C & ct) {
    FOR_UNROLL(ushort i = 0; i < ct.get_capacity(); ++i) {
        if (ct.is_valid_element(i)) {
            ct[i] = 0.f;
        }
    }
}

// One threadgroup's tile of the dense GEMM: the K loop and the store,
// shared by the plain and the segmented entries (gemm.metal's gemm_tile's
// place: `r0` the first weight row, `r1` the first activation row).
template <typename Q, typename T, bool RESID>
inline void tensor_tile(threadgroup char        * shmem,
                        device const typename Q::Block * W,
                        device const float      * x,
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
                        ushort sgitg) {
    constexpr int BK = gemm_bk<T>();
    constexpr int LDA = gemm_lda<T>();

    threadgroup T * sa = (threadgroup T *)shmem;
    threadgroup T * sb = sa + GEMM_ROWS * LDA;

    // The tile's live rows. The staging rows past them clamp to the last
    // live row — a defined access, never stored.
    const short nr0 = (short)min(GEMM_ROWS, rows - r0);
    const short nr1 = (short)min(GEMM_ACTS, acts - r1);
    const uint nb = (uint)cols / (uint)Q::values_per_block();

    // The thread's chunk of each side: row tiitg / 2, the K step's half
    // tiitg % 2.
    const short row = (short)(tiitg / 2);
    const short koff = (short)((tiitg % 2) * 16);
    device const typename Q::Block * a_src = W + (ulong)(r0 + min(row, (short)(nr0 - 1))) * nb;
    threadgroup vec<T, 4> * a_dst = (threadgroup vec<T, 4> *)(sa + row * LDA + koff);
    device const float * b_src = x + (ulong)(r1 + min(row, (short)(nr1 - 1))) * x_stride + koff;
    threadgroup vec<T, 4> * b_dst = (threadgroup vec<T, 4> *)(sb + row * LDA + koff);

    // The SIMD group's block: activation rows tm on, weight rows tn on.
    const short tm = TENSOR_SM * (short)(sgitg >> 1);
    const short tn = TENSOR_SN * (short)(sgitg & 1);
    const bool live = tm < nr1;

    tensor_mm<TENSOR_SM, TENSOR_SN> mm;
    auto ta = tensor_rows<T, TENSOR_SM>(sb + tm * LDA);
    auto tw = tensor_rows<T, TENSOR_SN>(sa + tn * LDA);
    auto ct = mm.template get_destination_cooperative_tensor<decltype(ta), decltype(tw), float>();
    tensor_zero(ct);

    for (int k0 = 0; k0 < cols; k0 += BK) {
        // The previous step's operands are read: the tiles may be written
        // over.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        gemm_stage_x<T>(b_src + k0, b_dst, faults, layer);
        tensor_stage_w<Q, T>(a_src, k0 + koff, a_dst);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (live) {
            mm.run(ta, tw, ct);
        }
    }

    // The store, per element under its row and column bound: no word past
    // the output's last row or column is written. The destination's index
    // is (weight row, activation row) in the block.
    if (!live) {
        return;
    }
    FOR_UNROLL(ushort i = 0; i < ct.get_capacity(); ++i) {
        if (!ct.is_valid_element(i)) {
            continue;
        }
        const auto at = ct.get_multidimensional_index(i);
        const short n = tn + (short)at[0];
        const short m = tm + (short)at[1];
        if (n < nr0 && m < nr1) {
            device float * p = y + (ulong)(r1 + m) * y_stride + (uint)(r0 + n);
            *p = RESID ? *p + ct[i] : ct[i];
        }
    }
}

// The plain GEMM, storing or adding into the output.
template <typename Q, typename T, bool RESID>
kernel void tensor_gemm_kernel(device const typename Q::Block * W [[buffer(0)]],
                               device const float      * x [[buffer(1)]],
                               device float            * y [[buffer(2)]],
                               constant GemmArgs       & args [[buffer(3)]],
                               device atomic_uint      * faults [[buffer(4)]],
                               uint3  tgpig [[threadgroup_position_in_grid]],
                               ushort tiitg [[thread_index_in_threadgroup]],
                               ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup char shmem[gemm_shmem_bytes<T>()];
    tensor_tile<Q, T, RESID>(shmem, W, x, y, (int)args.rows, (int)args.cols, (int)args.acts,
                             args.x_stride, args.y_stride, args.layer, faults,
                             (int)tgpig.y * GEMM_ROWS, (int)tgpig.x * GEMM_ACTS, tiitg, sgitg);
}

// The weight rows of up to four tensors of one type and one width
// (gemm.metal's gemm_seg_kernel's table and meaning).
template <typename Q, typename T, bool RESID>
kernel void tensor_gemm_seg_kernel(device const typename Q::Block * W [[buffer(0)]],
                                   device const float      * x [[buffer(1)]],
                                   device float            * y [[buffer(2)]],
                                   constant GemmSegArgs    & args [[buffer(3)]],
                                   device atomic_uint      * faults [[buffer(4)]],
                                   uint3  tgpig [[threadgroup_position_in_grid]],
                                   ushort tiitg [[thread_index_in_threadgroup]],
                                   ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup char shmem[gemm_shmem_bytes<T>()];
    uint s = 0;
    for (uint i = args.nseg; i > 0; --i) {
        if (tgpig.y >= args.segs[i - 1u].first_group) {
            s = i - 1u;
            break;
        }
    }
    const uint local = tgpig.y - args.segs[s].first_group;
    tensor_tile<Q, T, RESID>(shmem, W + (ulong)args.segs[s].wblocks, x, y + args.segs[s].out,
                             (int)args.segs[s].rows, (int)args.cols, (int)args.acts,
                             args.x_stride, args.y_stride, args.layer, faults,
                             (int)local * GEMM_ROWS, (int)tgpig.x * GEMM_ACTS, tiitg, sgitg);
}

// The pair's store of one block of places from `m0` on: element i of the
// gate's and the up's destinations is one output's, SwiGLU'd and weighted
// by its slot's route weight, under the run's length. `y_tile` is the
// tile's first place's row at the tile's first output.
template <typename C>
inline void tensor_pair_store(thread C & cg, thread C & cu, short m0, short tn, short count,
                              const device uint * places, device const float * route_w,
                              device float * y_tile, uint y_stride) {
    FOR_UNROLL(ushort i = 0; i < cg.get_capacity(); ++i) {
        if (!cg.is_valid_element(i)) {
            continue;
        }
        const auto at = cg.get_multidimensional_index(i);
        const short m = m0 + (short)at[1];
        if (m < count) {
            y_tile[(ulong)m * y_stride + (uint)(tn + (short)at[0])] =
                pair_swiglu(cg[i], cu[i]) * route_w[places[m]];
        }
    }
}

// The plain tile's store of one block of places from `m0` on, under the
// run's length.
template <typename C>
inline void tensor_plain_store(thread C & ct, short m0, short tn, short count,
                               device float * y_tile, uint y_stride) {
    FOR_UNROLL(ushort i = 0; i < ct.get_capacity(); ++i) {
        if (!ct.is_valid_element(i)) {
            continue;
        }
        const auto at = ct.get_multidimensional_index(i);
        const short m = m0 + (short)at[1];
        if (m < count) {
            y_tile[(ulong)m * y_stride + (uint)(tn + (short)at[0])] = ct[i];
        }
    }
}

// The grouped GEMM of the routed experts over the route table's tiles
// (gemm_routed.metal's gemm_routed_apart_kernel's tile, table and meaning, with one quant).
template <typename Q, typename T, bool PAIR>
kernel void tensor_routed_kernel(device const typename Q::Block * W [[buffer(0)]],
                                 device const typename Q::Block * W_up [[buffer(1)]],
                                 device const typename Q::Block * W_sh [[buffer(2)]],
                                 device const typename Q::Block * W_sh_up [[buffer(3)]],
                                 device const float      * x [[buffer(4)]],
                                 device float            * y [[buffer(5)]],
                                 device const uint4      * tiles [[buffer(6)]],
                                 device const uint       * sorted [[buffer(7)]],
                                 device const float      * route_w [[buffer(8)]],
                                 constant RoutedArgs     & args [[buffer(9)]],
                                 device atomic_uint      * faults [[buffer(10)]],
                                 uint2  tgpig [[threadgroup_position_in_grid]],
                                 ushort tiitg [[thread_index_in_threadgroup]],
                                 ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup char shmem[gemm_shmem_bytes<T>()];
    constexpr int BK = gemm_bk<T>();
    constexpr int LDA = gemm_lda<T>();

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
    device const typename Q::Block * wg = shared ? W_sh : W + (ulong)expert * args.stack_blocks;
    device const typename Q::Block * wu = PAIR ? (shared ? W_sh_up : W_up + (ulong)expert * args.stack_blocks) : wg;
    // The tile's first output: the pair's tile is 32 outputs (64 weight rows
    // of two tensors), the plain tile 64 rows of one.
    const uint r0 = tgpig.y * (PAIR ? ROUTED_ROWS / 2 : ROUTED_ROWS);
    const uint nb = args.cols / (uint)Q::values_per_block();

    threadgroup T * sa = (threadgroup T *)shmem;
    threadgroup T * sb = sa + ROUTED_ROWS * LDA;

    // A thread's chunk of each side: row tiitg / 2, the K step's half
    // tiitg % 2. The pair's staged rows: 0 to 31 the gate of outputs r0 on,
    // 32 to 63 their up.
    const short row = (short)(tiitg / 2);
    const short koff = (short)((tiitg % 2) * 16);
    device const typename Q::Block * a_src =
        PAIR ? (row < ROUTED_ROWS / 2 ? wg : wu) + (ulong)(r0 + (uint)(row % (ROUTED_ROWS / 2))) * nb
             : wg + (ulong)(r0 + (uint)row) * nb;
    threadgroup vec<T, 4> * a_dst = (threadgroup vec<T, 4> *)(sa + row * LDA + koff);

    // The activation rows the tile multiplies: the run's 16-row blocks. A
    // row past the run's length takes the run's last (a defined access).
    const short live_rows = TENSOR_RM * ((count + TENSOR_RM - 1) / TENSOR_RM);
    const bool b_stage = row < live_rows;
    const uint b_place = first + (uint)min(row, (short)(count - 1));
    const ulong src_row =
        args.gather != 0u ? (ulong)(sorted[b_place] / BLOOMERY_EXPERT_SLOTS) : (ulong)b_place;
    device const float * b_src = x + src_row * args.x_stride + koff;
    threadgroup vec<T, 4> * b_dst = (threadgroup vec<T, 4> *)(sb + row * LDA + koff);

    // The SIMD group's two blocks of places, tm and tm + TENSOR_RM on, each
    // live when the run reaches it.
    const short tm = TENSOR_SM * (short)(sgitg >> 1);
    const bool live0 = tm < count;
    const bool live1 = tm + TENSOR_RM < count;
    auto ta0 = tensor_rows<T, TENSOR_RM>(sb + tm * LDA);
    auto ta1 = tensor_rows<T, TENSOR_RM>(sb + (tm + TENSOR_RM) * LDA);
    const device uint * places = sorted + first;
    device float * y_tile = y + (ulong)first * args.y_stride + r0;

    if (PAIR) {
        // The group's 16 outputs' gate rows and up rows: one op type, so the
        // two destinations share one layout.
        const short tn = TENSOR_PAIR_N * (short)(sgitg & 1);
        tensor_mm<TENSOR_RM, TENSOR_PAIR_N> mm;
        auto tg = tensor_rows<T, TENSOR_PAIR_N>(sa + tn * LDA);
        auto tu = tensor_rows<T, TENSOR_PAIR_N>(sa + (ROUTED_ROWS / 2 + tn) * LDA);
        auto cg0 = mm.template get_destination_cooperative_tensor<decltype(ta0), decltype(tg), float>();
        auto cu0 = mm.template get_destination_cooperative_tensor<decltype(ta0), decltype(tu), float>();
        auto cg1 = mm.template get_destination_cooperative_tensor<decltype(ta1), decltype(tg), float>();
        auto cu1 = mm.template get_destination_cooperative_tensor<decltype(ta1), decltype(tu), float>();
        tensor_zero(cg0);
        tensor_zero(cu0);
        tensor_zero(cg1);
        tensor_zero(cu1);
        for (int k0 = 0; k0 < (int)args.cols; k0 += BK) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (b_stage) {
                gemm_stage_x<T>(b_src + k0, b_dst, faults, args.layer);
            }
            tensor_stage_w<Q, T>(a_src, k0 + koff, a_dst);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (live0) {
                mm.run(ta0, tg, cg0);
                mm.run(ta0, tu, cu0);
            }
            if (live1) {
                mm.run(ta1, tg, cg1);
                mm.run(ta1, tu, cu1);
            }
        }
        if (live0) {
            tensor_pair_store(cg0, cu0, tm, tn, count, places, route_w, y_tile, args.y_stride);
        }
        if (live1) {
            tensor_pair_store(cg1, cu1, tm + TENSOR_RM, tn, count, places, route_w, y_tile,
                              args.y_stride);
        }
    } else {
        const short tn = TENSOR_SN * (short)(sgitg & 1);
        tensor_mm<TENSOR_RM, TENSOR_SN> mm;
        auto tw = tensor_rows<T, TENSOR_SN>(sa + tn * LDA);
        auto c0 = mm.template get_destination_cooperative_tensor<decltype(ta0), decltype(tw), float>();
        auto c1 = mm.template get_destination_cooperative_tensor<decltype(ta1), decltype(tw), float>();
        tensor_zero(c0);
        tensor_zero(c1);
        for (int k0 = 0; k0 < (int)args.cols; k0 += BK) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (b_stage) {
                gemm_stage_x<T>(b_src + k0, b_dst, faults, args.layer);
            }
            tensor_stage_w<Q, T>(a_src, k0 + koff, a_dst);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (live0) {
                mm.run(ta0, tw, c0);
            }
            if (live1) {
                mm.run(ta1, tw, c1);
            }
        }
        if (live0) {
            tensor_plain_store(c0, tm, tn, count, y_tile, args.y_stride);
        }
        if (live1) {
            tensor_plain_store(c1, tm + TENSOR_RM, tn, count, y_tile, args.y_stride);
        }
    }
}

// The entries, under the simdgroup kernels' host names.

#define TENSOR_GEMM_ENTRY(SUFFIX, Q, T, RESID, SEG, ARGS)                             \
    template [[host_name("gemm_" #SUFFIX)]] kernel void SEG<Q, T, RESID>(             \
        device const Q::Block * W [[buffer(0)]],                                     \
        device const float    * x [[buffer(1)]],                                     \
        device float          * y [[buffer(2)]],                                     \
        constant ARGS         & args [[buffer(3)]],                                  \
        device atomic_uint    * faults [[buffer(4)]],                                \
        uint3  tgpig [[threadgroup_position_in_grid]],                               \
        ushort tiitg [[thread_index_in_threadgroup]],                                \
        ushort sgitg [[simdgroup_index_in_threadgroup]]);

#define TENSOR_GEMM_FAMILY(STEM, Q)                                                                   \
    TENSOR_GEMM_ENTRY(STEM##_store_f16, Q, half, false, tensor_gemm_kernel, GemmArgs)                 \
    TENSOR_GEMM_ENTRY(STEM##_store_f32, Q, float, false, tensor_gemm_kernel, GemmArgs)                \
    TENSOR_GEMM_ENTRY(STEM##_resid_f16, Q, half, true, tensor_gemm_kernel, GemmArgs)                  \
    TENSOR_GEMM_ENTRY(STEM##_resid_f32, Q, float, true, tensor_gemm_kernel, GemmArgs)                 \
    TENSOR_GEMM_ENTRY(STEM##_seg_store_f16, Q, half, false, tensor_gemm_seg_kernel, GemmSegArgs)      \
    TENSOR_GEMM_ENTRY(STEM##_seg_store_f32, Q, float, false, tensor_gemm_seg_kernel, GemmSegArgs)     \
    TENSOR_GEMM_ENTRY(STEM##_seg_resid_f16, Q, half, true, tensor_gemm_seg_kernel, GemmSegArgs)       \
    TENSOR_GEMM_ENTRY(STEM##_seg_resid_f32, Q, float, true, tensor_gemm_seg_kernel, GemmSegArgs)

TENSOR_GEMM_FAMILY(q4k, Q4K)
TENSOR_GEMM_FAMILY(q6k, Q6K)
TENSOR_GEMM_FAMILY(a4, A4)
TENSOR_GEMM_FAMILY(a8, A8)
TENSOR_GEMM_FAMILY(a4h, A4H)
TENSOR_GEMM_FAMILY(a8h, A8H)
TENSOR_GEMM_FAMILY(q8_0, Q8_0)
TENSOR_GEMM_FAMILY(q5k, Q5K)

#define TENSOR_ROUTED_ENTRY(NAME, Q, T, PAIR)                                       \
    template [[host_name("gemm_routed_" #NAME)]]                                    \
    kernel void tensor_routed_kernel<Q, T, PAIR>(                                   \
        device const Q::Block * W [[buffer(0)]],                                    \
        device const Q::Block * W_up [[buffer(1)]],                                 \
        device const Q::Block * W_sh [[buffer(2)]],                                 \
        device const Q::Block * W_sh_up [[buffer(3)]],                              \
        device const float    * x [[buffer(4)]],                                    \
        device float          * y [[buffer(5)]],                                    \
        device const uint4    * tiles [[buffer(6)]],                                \
        device const uint     * sorted [[buffer(7)]],                               \
        device const float    * route_w [[buffer(8)]],                              \
        constant RoutedArgs   & args [[buffer(9)]],                                 \
        device atomic_uint    * faults [[buffer(10)]],                              \
        uint2  tgpig [[threadgroup_position_in_grid]],                              \
        ushort tiitg [[thread_index_in_threadgroup]],                               \
        ushort sgitg [[simdgroup_index_in_threadgroup]]);

TENSOR_ROUTED_ENTRY(q4k_pair_f16, Q4K, half, true)
TENSOR_ROUTED_ENTRY(q4k_pair_f32, Q4K, float, true)
TENSOR_ROUTED_ENTRY(q4k_plain_f16, Q4K, half, false)
TENSOR_ROUTED_ENTRY(q4k_plain_f32, Q4K, float, false)
TENSOR_ROUTED_ENTRY(q6k_pair_f16, Q6K, half, true)
TENSOR_ROUTED_ENTRY(q6k_pair_f32, Q6K, float, true)
TENSOR_ROUTED_ENTRY(q6k_plain_f16, Q6K, half, false)
TENSOR_ROUTED_ENTRY(q6k_plain_f32, Q6K, float, false)
TENSOR_ROUTED_ENTRY(a4_pair_f16, A4, half, true)
TENSOR_ROUTED_ENTRY(a4_pair_f32, A4, float, true)
TENSOR_ROUTED_ENTRY(a4_plain_f16, A4, half, false)
TENSOR_ROUTED_ENTRY(a4_plain_f32, A4, float, false)
TENSOR_ROUTED_ENTRY(a8_pair_f16, A8, half, true)
TENSOR_ROUTED_ENTRY(a8_pair_f32, A8, float, true)
TENSOR_ROUTED_ENTRY(a8_plain_f16, A8, half, false)
TENSOR_ROUTED_ENTRY(a8_plain_f32, A8, float, false)
TENSOR_ROUTED_ENTRY(a4h_pair_f16, A4H, half, true)
TENSOR_ROUTED_ENTRY(a4h_pair_f32, A4H, float, true)
TENSOR_ROUTED_ENTRY(a4h_plain_f16, A4H, half, false)
TENSOR_ROUTED_ENTRY(a4h_plain_f32, A4H, float, false)
TENSOR_ROUTED_ENTRY(a8h_pair_f16, A8H, half, true)
TENSOR_ROUTED_ENTRY(a8h_pair_f32, A8H, float, true)
TENSOR_ROUTED_ENTRY(a8h_plain_f16, A8H, half, false)
TENSOR_ROUTED_ENTRY(a8h_plain_f32, A8H, float, false)
TENSOR_ROUTED_ENTRY(q8_0_pair_f16, Q8_0, half, true)
TENSOR_ROUTED_ENTRY(q8_0_pair_f32, Q8_0, float, true)
TENSOR_ROUTED_ENTRY(q8_0_plain_f16, Q8_0, half, false)
TENSOR_ROUTED_ENTRY(q8_0_plain_f32, Q8_0, float, false)
TENSOR_ROUTED_ENTRY(q5k_pair_f16, Q5K, half, true)
TENSOR_ROUTED_ENTRY(q5k_pair_f32, Q5K, float, true)
TENSOR_ROUTED_ENTRY(q5k_plain_f16, Q5K, half, false)
TENSOR_ROUTED_ENTRY(q5k_plain_f32, Q5K, float, false)
