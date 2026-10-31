// The routed expert gate-up pair and the folded down. One dispatch each,
// one body per shape over the quant (quant.metal), the entries the
// instantiations under their host names:
//
//   q4k_pair_swiglu   y[slot, r] = silu(gate_r · x̂) * (up_r · x̂),
//                     x̂ = x * gain / sqrt(mean(x^2) + eps)
//   q6k_pair_swiglu   the same over block_q6_K
//   a4_pair_swiglu    the same over block_mlx_a4
//   a8_pair_swiglu    the same over block_mlx_a8
//   a4h_pair_swiglu   the same over block_mlx_a4 with F16 scales and biases
//   a8h_pair_swiglu   the same over block_mlx_a8 with F16 scales and biases
//   q8_0_pair_swiglu  the same over block_q8_0
//   q5k_pair_swiglu   the same over block_q5_K
//   q4k_down_fold     y[r] += sum over slots of w_s * (down_s[r] · h_s)
//   q6k_down_fold     the same over block_q6_K
//   a4_down_fold      the same over block_mlx_a4
//   a8_down_fold      the same over block_mlx_a8
//   a4h_down_fold     the same over block_mlx_a4 with F16 scales and biases
//   a8h_down_fold     the same over block_mlx_a8 with F16 scales and biases
//   q8_0_down_fold    the same over block_q8_0
//   q5k_down_fold     the same over block_q5_K
//
// The bodies' dot loops are llama.cpp's (kernel_mul_mv_q4_K_f32,
// kernel_mul_mv_q6_K_f32, kernel_mul_mv_q8_0_f32 and kernel_mul_mv_q5_K_f32,
// MIT licence; see THIRD_PARTY_NOTICES.md):
//   ggml/src/ggml-metal/kernels/mul_mv.metal
// The lane map, the block loop and the simd_sum reduction are the ones the
// dense mat-vec uses, owned by the quant.
//
// The pair tile is one pair of weight rows (a gate row and the up row of one
// output) per two rows of the mat-vec geometry: a SIMD group owns
// BLOOMERY_MATVEC_ROWS_PER_SIMD_GROUP weight rows, half that many outputs, and
// the threadgroup holds BLOOMERY_MATVEC_SIMD_GROUPS of them. The tile is the
// dense mat-vec's register shape: the group loads its share of the input once
// and dots it with every row it owns. grid.y is the slot. Slot 8 reads the
// shared-expert tensors; every other slot reads stack + ids[slot] * expert
// bytes. The id is an int32, then that expert's rows, the way ggml mul_mv_id
// reads an id. The select is uniform in the threadgroup. The route weight is
// not applied here.
//
// The pair norms its input as it loads it, the dense mat-vec's norm on load
// (norm.metal) in its order: every SIMD group loads the whole input once, the
// lane adds the squares of what it loaded before the gain multiplies them,
// the SIMD group sums the squares, and the scale multiplies both finished
// dots. Every slot, the shared one too, dots with that one normed input. A
// norm that cannot take its input raises NormQuant and writes NaN in the
// row, as the mat-vec's does.
//
// The fold's K' is nine slots of K 512: eighteen 256-value blocks,
// seventy-two of MLX's 64-value groups, or a hundred and forty-four Q8_0
// blocks — the quant's own count. A lane owns blocks by the dense stride
// across that whole K', not slot by slot: Q4_K and Q5_K step four blocks (a
// lane owns four or five), Q6_K steps two (a lane owns nine), A4, A8, A4H,
// A8H and Q8_0 step eight (nine and eighteen). The route weight of the block's slot
// scales the activation as it is loaded, and the residual add is the store.
// There is no second dispatch.
//
// The pair is one template over two quants, the stacks' and the shared
// expert's, `pair_swiglu_apart_kernel<Q, QS>`. A file whose shared expert has
// the stacks' type instantiates it with `QS == Q`, under the plain host names;
// a shared expert of another type (`Apart`, the Q8_0 shared expert of a
// mixed-type file) takes the `*_apart_*` entries. The pair's slot is uniform
// in a threadgroup, so the shared slot's threadgroups run the shared quant's
// body whole. The fold has a kernel for each case: the same type
// (`down_fold_kernel<Q>`) and two quants (`down_fold_apart_kernel<Q, QS>`),
// whose K' is two ranges, the eight routed slots' blocks in the stacks' quant
// and the shared slot's in its own; a lane walks each range by its quant's
// stride and the SIMD group sums both into one row.
//
// An id past the stack raises ExpertId and writes NaN before any weight is
// loaded. The pair writes NaN only in the bad slot's row; the fold writes NaN
// in every row of the threadgroup, because every row reads every slot.
//
// This file comes after matvec_q6k.metal and quant.metal (the decode owners),
// norm.metal (rms_scale, rms_usable), gemm_shared.metal (pair_swiglu) and
// fault.metal (raise).

#include <metal_stdlib>

using namespace metal;

#pragma METAL fp contract(off)

// The outputs one SIMD group of a pair owns: the mat-vec geometry's rows, a
// gate row and an up row to an output.
#define PAIR_OUTS (BLOOMERY_MATVEC_ROWS_PER_SIMD_GROUP / 2)

static_assert(BLOOMERY_MATVEC_ROWS_PER_SIMD_GROUP % 2 == 0,
              "a pair's weight rows are a gate and an up");

// Six words: the output rows, the blocks of one row, the experts in the
// stack, the layer a raise names, the output stride in floats, and the
// norm's epsilon.
struct PairArgs {
    uint rows;
    uint nblocks;
    uint n_experts;
    uint layer;
    uint stride;
    float eps;
};

static_assert(sizeof(PairArgs) == 24, "PairArgs is six words");

// Four words: the output rows, the experts in the stack, the layer, and a pad
// so the block is four words. The fold's K' is fixed: the host refuses any
// other expert width, and the kernel owns eighteen blocks.
struct FoldArgs {
    uint rows;
    uint n_experts;
    uint layer;
    uint pad;
};

static_assert(sizeof(FoldArgs) == 16, "FoldArgs is four words");

// The shared slot reads `shexp`; every other slot reads `stack`. One copy of
// the select, so every quant of the fold follows it.
inline device const char * expert_base(bool shared_slot,
                                       device const char * shexp,
                                       device const char * stack) {
    return shared_slot ? shexp : stack;
}

// True when the routed slot's id is past the stack. Slot 8 is the shared
// expert and leaves `expert` at 0; the caller reads the shared tensors.
inline bool pair_expert(device const int * ids, uint slot, uint n_experts,
                        thread int & expert) {
    expert = 0;
    if (slot == BLOOMERY_EXPERT_SHARED) {
        return false;
    }
    int id = ids[slot];
    if (id < 0 || (uint)id >= n_experts) {
        return true;
    }
    expert = id;
    return false;
}

// True when every routed id is inside the stack. The shared slot is not an id.
inline bool fold_ids_ok(device const int * ids, uint n_experts) {
    for (uint s = 0u; s < BLOOMERY_EXPERT_ROUTED; ++s) {
        int id = ids[s];
        if (id < 0 || (uint)id >= n_experts) {
            return false;
        }
    }
    return true;
}

// A slot's rows the pair cannot compute: `site` raised in the layer, NaN in
// every row of the threadgroup's.
inline void pair_write_bad(device float * y, uint slot, uint stride, int row0, int rows,
                           device atomic_uint * faults, uint layer, uint site) {
    raise(faults, layer, site);
    for (short o = 0; o < PAIR_OUTS && row0 + o < rows; ++o) {
        y[(ulong)slot * stride + (uint)(row0 + o)] = NAN;
    }
}

inline void fold_write_bad(device float * y, int row0, int rows,
                           device atomic_uint * faults, uint layer) {
    raise(faults, layer, BLOOMERY_FAULT_SITE_EXPERT_ID);
    for (short r = 0; r < BLOOMERY_MATVEC_ROWS_PER_SIMD_GROUP && row0 + r < rows; ++r) {
        y[(uint)(row0 + r)] = NAN;
    }
}

// One block of the folded K'. The slot owns `per_slot` blocks — the quant's
// count for one expert width — and the shared slot is the shared tensor. The
// stack is not contiguous across experts, so the pointer is this block's own.
template <typename Q>
inline device const typename Q::Block * fold_block(device const typename Q::Block * stack,
                                                   device const typename Q::Block * shared,
                                                   device const int * ids,
                                                   uint rows,
                                                   int row,
                                                   int ib) {
    const int per_slot = BLOOMERY_EXPERT_K / Q::values_per_block();
    int slot = ib / per_slot;
    int local = ib % per_slot;
    bool shared_slot = slot == BLOOMERY_EXPERT_SHARED;
    int expert = shared_slot ? 0 : ids[slot];
    device const char * base = expert_base(shared_slot,
                                           (device const char *)shared,
                                           (device const char *)stack);
    ulong index = ((ulong)expert * (ulong)rows + (ulong)row) * (ulong)per_slot + (ulong)local;
    return ((device const typename Q::Block *)base) + index;
}

template <typename Q>
kernel void down_fold_kernel(device const typename Q::Block * down [[buffer(0)]],
                             device const typename Q::Block * shared [[buffer(1)]],
                             device const float * h [[buffer(2)]],
                             device const float * w [[buffer(3)]],
                             device const int * ids [[buffer(4)]],
                             device float * y [[buffer(5)]],
                             device atomic_uint * faults [[buffer(6)]],
                             constant FoldArgs & args [[buffer(7)]],
                             uint3 tgpig [[threadgroup_position_in_grid]],
                             ushort tiisg [[thread_index_in_simdgroup]],
                             ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    typename Q::Lane lane = Q::lane_of(tiisg);
    // BLOOMERY_MATVEC_ROWS_PER_SIMD_GROUP rows per SIMD group,
    // BLOOMERY_MATVEC_SIMD_GROUPS of them in the threadgroup.
    constexpr short NR0 = BLOOMERY_MATVEC_ROWS_PER_SIMD_GROUP;
    const int row0 = (int)((tgpig.x * BLOOMERY_MATVEC_SIMD_GROUPS + (uint)sgitg) * (uint)NR0);
    if (row0 >= (int)args.rows) {
        return;
    }

    // Every row reads every slot, so one bad id condemns the whole row, and
    // the check finishes before any weight pointer is formed.
    if (!fold_ids_ok(ids, args.n_experts)) {
        if (tiisg == 0u) {
            fold_write_bad(y, row0, (int)args.rows, faults, args.layer);
        }
        return;
    }

    float sum[NR0];
    for (short r = 0; r < NR0; ++r) {
        sum[r] = 0.f;
    }
    for (int ib = Q::first_block(lane); ib < Q::fold_blocks(); ib += Q::block_stride()) {
        float scale = w[Q::fold_slot(ib)];
        Q::template load<LOAD_ROUTE>(lane, h + (ulong)ib * Q::values_per_block(), nullptr,
                                     scale, nullptr);
        for (short r = 0; r < NR0 && row0 + r < (int)args.rows; ++r) {
            device const typename Q::Block * b =
                fold_block<Q>(down, shared, ids, args.rows, row0 + r, ib);
            sum[r] += Q::partial(*b, lane);
        }
    }
    for (short r = 0; r < NR0 && row0 + r < (int)args.rows; ++r) {
        float d = simd_sum(sum[r]);
        if (tiisg == 0u) {
            y[(uint)(row0 + r)] += d;
        }
    }
}

// The dots of one pair slot over quant Q: `gate` and `up` are the slot's
// expert tensors, `nblocks` the blocks of one of their rows. The pair kernel
// runs it for a routed slot over the stacks' quant and for the shared slot
// over the shared expert's. SCALED: `x` is the join's x · gain and `gain`
// points at that join's scale word (matvec_rows' SCALED): the load is the
// plain one and the dots are scaled as the norm's, the same bits.
template <typename Q, bool SCALED = false>
inline void pair_slot(device const typename Q::Block * gate,
                      device const typename Q::Block * up,
                      device const float * x,
                      device const float * gain,
                      device float * y,
                      device atomic_uint * faults,
                      constant PairArgs & args,
                      uint nblocks,
                      uint slot,
                      int row0,
                      ushort tiisg) {
    typename Q::Lane lane = Q::lane_of(tiisg);
    device const typename Q::Block * grow[PAIR_OUTS];
    device const typename Q::Block * urow[PAIR_OUTS];
    for (short o = 0; o < PAIR_OUTS; ++o) {
        grow[o] = gate + (ulong)(row0 + o) * (ulong)nblocks;
        urow[o] = up + (ulong)(row0 + o) * (ulong)nblocks;
    }
    float sumg[PAIR_OUTS];
    float sumu[PAIR_OUTS];
    for (short o = 0; o < PAIR_OUTS; ++o) {
        sumg[o] = 0.f;
        sumu[o] = 0.f;
    }
    float sum_sq = 0.f;
    // The join's scale, read before the walk so its load is in flight under it.
    const float in_scale = SCALED ? gain[0] : 1.f;
    for (int ib = Q::first_block(lane); ib < (int)nblocks; ib += Q::block_stride()) {
        if (SCALED) {
            Q::template load<LOAD_PLAIN>(lane, x + (ulong)ib * Q::values_per_block(), nullptr,
                                         1.f, nullptr);
        } else {
            Q::template load<LOAD_NORM>(lane, x + (ulong)ib * Q::values_per_block(),
                                        gain + (ulong)ib * Q::values_per_block(), 1.f, &sum_sq);
        }
        for (short o = 0; o < PAIR_OUTS; ++o) {
            sumg[o] += Q::partial(grow[o][ib], lane);
            sumu[o] += Q::partial(urow[o][ib], lane);
        }
    }
    const float sum_squares = SCALED ? 0.f : simd_sum(sum_sq);
    const float scale =
        SCALED ? in_scale : rms_scale(sum_squares, nblocks * Q::values_per_block(), args.eps);
    if (!SCALED && !rms_usable(sum_squares, scale)) {
        if (tiisg == 0u) {
            pair_write_bad(y, slot, args.stride, row0, (int)args.rows, faults, args.layer,
                           BLOOMERY_FAULT_SITE_NORM_QUANT);
        }
        return;
    }
    for (short o = 0; o < PAIR_OUTS && row0 + o < (int)args.rows; ++o) {
        float g = simd_sum(sumg[o]) * scale;
        float u = simd_sum(sumu[o]) * scale;
        if (tiisg == 0u) {
            y[(ulong)slot * args.stride + (uint)(row0 + o)] = pair_swiglu(g, u);
        }
    }
}

// The pair over stacks of quant Q and a shared expert of quant QS. `nblocks`
// in the arguments is the stacks' rows' blocks; the shared expert's rows hold
// the same columns in QS's blocks, as many when QS is Q.
template <typename Q, typename QS, bool SCALED = false>
kernel void pair_swiglu_apart_kernel(device const typename Q::Block * gate [[buffer(0)]],
                                     device const typename Q::Block * up [[buffer(1)]],
                                     device const typename QS::Block * shexp_gate [[buffer(2)]],
                                     device const typename QS::Block * shexp_up [[buffer(3)]],
                                     device const float * x [[buffer(4)]],
                                     device const float * gain [[buffer(5)]],
                                     device const int * ids [[buffer(6)]],
                                     device float * y [[buffer(7)]],
                                     device atomic_uint * faults [[buffer(8)]],
                                     constant PairArgs & args [[buffer(9)]],
                                     uint3 tgpig [[threadgroup_position_in_grid]],
                                     ushort tiisg [[thread_index_in_simdgroup]],
                                     ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint slot = tgpig.y;
    if (slot >= BLOOMERY_EXPERT_SLOTS) {
        return;
    }
    const int row0 = (int)((tgpig.x * BLOOMERY_MATVEC_SIMD_GROUPS + (uint)sgitg) * (uint)PAIR_OUTS);
    if (row0 >= (int)args.rows) {
        return;
    }

    int expert = 0;
    if (pair_expert(ids, slot, args.n_experts, expert)) {
        if (tiisg == 0u) {
            pair_write_bad(y, slot, args.stride, row0, (int)args.rows, faults, args.layer,
                           BLOOMERY_FAULT_SITE_EXPERT_ID);
        }
        return;
    }
    if (slot == BLOOMERY_EXPERT_SHARED) {
        const uint shared_blocks = args.nblocks * (uint)Q::values_per_block() / (uint)QS::values_per_block();
        pair_slot<QS, SCALED>(shexp_gate, shexp_up, x, gain, y, faults, args, shared_blocks, slot,
                              row0, tiisg);
    } else {
        const ulong expert_blocks = (ulong)expert * (ulong)args.rows * (ulong)args.nblocks;
        pair_slot<Q, SCALED>(gate + expert_blocks, up + expert_blocks, x, gain, y, faults, args,
                             args.nblocks, slot, row0, tiisg);
    }
}

// The eight routed slots' share of the fold's rows over quant Q: the lane's
// blocks of the routed range, the route weight of the block's slot scaling
// the activation as it is loaded.
template <typename Q>
inline void fold_routed(thread float * sum,
                        device const typename Q::Block * down,
                        device const float * h,
                        device const float * w,
                        device const int * ids,
                        uint rows,
                        int row0,
                        ushort tiisg) {
    constexpr short NR0 = BLOOMERY_MATVEC_ROWS_PER_SIMD_GROUP;
    const int per_slot = BLOOMERY_EXPERT_K / Q::values_per_block();
    typename Q::Lane lane = Q::lane_of(tiisg);
    for (int ib = Q::first_block(lane); ib < BLOOMERY_EXPERT_ROUTED * per_slot; ib += Q::block_stride()) {
        float scale = w[ib / per_slot];
        Q::template load<LOAD_ROUTE>(lane, h + (ulong)ib * Q::values_per_block(), nullptr,
                                     scale, nullptr);
        for (short r = 0; r < NR0 && row0 + r < (int)rows; ++r) {
            device const typename Q::Block * b =
                fold_block<Q>(down, down, ids, rows, row0 + r, ib);
            sum[r] += Q::partial(*b, lane);
        }
    }
}

// The shared slot's share over quant QS: its blocks of the shared tensor,
// the slot's route weight scaling the activation.
template <typename QS>
inline void fold_shared(thread float * sum,
                        device const typename QS::Block * shared,
                        device const float * h,
                        device const float * w,
                        uint rows,
                        int row0,
                        ushort tiisg) {
    constexpr short NR0 = BLOOMERY_MATVEC_ROWS_PER_SIMD_GROUP;
    const int per_slot = BLOOMERY_EXPERT_K / QS::values_per_block();
    typename QS::Lane lane = QS::lane_of(tiisg);
    const float scale = w[BLOOMERY_EXPERT_SHARED];
    device const float * hs = h + (ulong)BLOOMERY_EXPERT_SHARED * BLOOMERY_EXPERT_K;
    for (int ib = QS::first_block(lane); ib < per_slot; ib += QS::block_stride()) {
        QS::template load<LOAD_ROUTE>(lane, hs + (ulong)ib * QS::values_per_block(), nullptr,
                                      scale, nullptr);
        for (short r = 0; r < NR0 && row0 + r < (int)rows; ++r) {
            sum[r] += QS::partial(shared[(ulong)(row0 + r) * (ulong)per_slot + (ulong)ib], lane);
        }
    }
}

// The fold over a down stack of quant Q and a shared down of quant QS.
template <typename Q, typename QS>
kernel void down_fold_apart_kernel(device const typename Q::Block * down [[buffer(0)]],
                                   device const typename QS::Block * shared [[buffer(1)]],
                                   device const float * h [[buffer(2)]],
                                   device const float * w [[buffer(3)]],
                                   device const int * ids [[buffer(4)]],
                                   device float * y [[buffer(5)]],
                                   device atomic_uint * faults [[buffer(6)]],
                                   constant FoldArgs & args [[buffer(7)]],
                                   uint3 tgpig [[threadgroup_position_in_grid]],
                                   ushort tiisg [[thread_index_in_simdgroup]],
                                   ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr short NR0 = BLOOMERY_MATVEC_ROWS_PER_SIMD_GROUP;
    const int row0 = (int)((tgpig.x * BLOOMERY_MATVEC_SIMD_GROUPS + (uint)sgitg) * (uint)NR0);
    if (row0 >= (int)args.rows) {
        return;
    }
    if (!fold_ids_ok(ids, args.n_experts)) {
        if (tiisg == 0u) {
            fold_write_bad(y, row0, (int)args.rows, faults, args.layer);
        }
        return;
    }

    float sum[NR0];
    for (short r = 0; r < NR0; ++r) {
        sum[r] = 0.f;
    }
    fold_routed<Q>(sum, down, h, w, ids, args.rows, row0, tiisg);
    fold_shared<QS>(sum, shared, h, w, args.rows, row0, tiisg);
    for (short r = 0; r < NR0 && row0 + r < (int)args.rows; ++r) {
        const float total = simd_sum(sum[r]);
        if (tiisg == 0u) {
            y[(uint)(row0 + r)] += total;
        }
    }
}

#define APART_PAIR_ENTRY(NAME, Q, BQ, QS, BQS)                                         \
    template [[host_name(NAME)]]                                                       \
    kernel void pair_swiglu_apart_kernel<Q, QS>(                                       \
        device const BQ * gate [[buffer(0)]],                                          \
        device const BQ * up [[buffer(1)]],                                            \
        device const BQS * shexp_gate [[buffer(2)]],                                   \
        device const BQS * shexp_up [[buffer(3)]],                                     \
        device const float * x [[buffer(4)]],                                          \
        device const float * gain [[buffer(5)]],                                       \
        device const int * ids [[buffer(6)]],                                          \
        device float * y [[buffer(7)]],                                                \
        device atomic_uint * faults [[buffer(8)]],                                     \
        constant PairArgs & args [[buffer(9)]],                                        \
        uint3 tgpig [[threadgroup_position_in_grid]],                                  \
        ushort tiisg [[thread_index_in_simdgroup]],                                    \
        ushort sgitg [[simdgroup_index_in_threadgroup]]);

#define APART_FOLD_ENTRY(NAME, Q, BQ, QS, BQS)                                         \
    template [[host_name(NAME)]]                                                       \
    kernel void down_fold_apart_kernel<Q, QS>(                                         \
        device const BQ * down [[buffer(0)]],                                          \
        device const BQS * shared [[buffer(1)]],                                       \
        device const float * h [[buffer(2)]],                                          \
        device const float * w [[buffer(3)]],                                          \
        device const int * ids [[buffer(4)]],                                          \
        device float * y [[buffer(5)]],                                                \
        device atomic_uint * faults [[buffer(6)]],                                     \
        constant FoldArgs & args [[buffer(7)]],                                        \
        uint3 tgpig [[threadgroup_position_in_grid]],                                  \
        ushort tiisg [[thread_index_in_simdgroup]],                                    \
        ushort sgitg [[simdgroup_index_in_threadgroup]]);

// The pair of a shared expert in the stacks' type, per quant.
APART_PAIR_ENTRY("q4k_pair_swiglu", Q4K, block_q4_K, Q4K, block_q4_K)
APART_PAIR_ENTRY("q6k_pair_swiglu", Q6K, block_q6_K, Q6K, block_q6_K)
APART_PAIR_ENTRY("a4_pair_swiglu", A4, block_mlx_a4, A4, block_mlx_a4)
APART_PAIR_ENTRY("a8_pair_swiglu", A8, block_mlx_a8, A8, block_mlx_a8)
APART_PAIR_ENTRY("a4h_pair_swiglu", A4H, block_mlx_a4, A4H, block_mlx_a4)
APART_PAIR_ENTRY("a8h_pair_swiglu", A8H, block_mlx_a8, A8H, block_mlx_a8)
APART_PAIR_ENTRY("q8_0_pair_swiglu", Q8_0, block_q8_0, Q8_0, block_q8_0)
APART_PAIR_ENTRY("q5k_pair_swiglu", Q5K, block_q5_K, Q5K, block_q5_K)

// The scaled pair of each type, a shared expert in the stacks' type: the
// input is the join's x · gain, `gain` binds the join's scale word.
#define SCALED_PAIR_ENTRY(NAME, Q, BQ)                                                 \
    template [[host_name(NAME)]]                                                       \
    kernel void pair_swiglu_apart_kernel<Q, Q, true>(                                  \
        device const BQ * gate [[buffer(0)]],                                          \
        device const BQ * up [[buffer(1)]],                                            \
        device const BQ * shexp_gate [[buffer(2)]],                                    \
        device const BQ * shexp_up [[buffer(3)]],                                      \
        device const float * x [[buffer(4)]],                                          \
        device const float * gain [[buffer(5)]],                                       \
        device const int * ids [[buffer(6)]],                                          \
        device float * y [[buffer(7)]],                                                \
        device atomic_uint * faults [[buffer(8)]],                                     \
        constant PairArgs & args [[buffer(9)]],                                        \
        uint3 tgpig [[threadgroup_position_in_grid]],                                  \
        ushort tiisg [[thread_index_in_simdgroup]],                                    \
        ushort sgitg [[simdgroup_index_in_threadgroup]]);

SCALED_PAIR_ENTRY("q4k_pair_swiglu_scaled", Q4K, block_q4_K)
SCALED_PAIR_ENTRY("q6k_pair_swiglu_scaled", Q6K, block_q6_K)
SCALED_PAIR_ENTRY("a4_pair_swiglu_scaled", A4, block_mlx_a4)
SCALED_PAIR_ENTRY("a8_pair_swiglu_scaled", A8, block_mlx_a8)
SCALED_PAIR_ENTRY("a4h_pair_swiglu_scaled", A4H, block_mlx_a4)
SCALED_PAIR_ENTRY("a8h_pair_swiglu_scaled", A8H, block_mlx_a8)
SCALED_PAIR_ENTRY("q8_0_pair_swiglu_scaled", Q8_0, block_q8_0)
SCALED_PAIR_ENTRY("q5k_pair_swiglu_scaled", Q5K, block_q5_K)

// The pairs of types a mixed file has: Q4_K and Q5_K gate and up stacks, Q5_K
// and Q6_K down stacks, each beside a Q8_0 shared expert.
APART_PAIR_ENTRY("q4k_pair_swiglu_apart_q8_0", Q4K, block_q4_K, Q8_0, block_q8_0)
APART_PAIR_ENTRY("q5k_pair_swiglu_apart_q8_0", Q5K, block_q5_K, Q8_0, block_q8_0)
APART_FOLD_ENTRY("q5k_down_fold_apart_q8_0", Q5K, block_q5_K, Q8_0, block_q8_0)
APART_FOLD_ENTRY("q6k_down_fold_apart_q8_0", Q6K, block_q6_K, Q8_0, block_q8_0)

template [[host_name("q4k_down_fold")]]
kernel void down_fold_kernel<Q4K>(device const block_q4_K * down [[buffer(0)]],
                                  device const block_q4_K * shared [[buffer(1)]],
                                  device const float * h [[buffer(2)]],
                                  device const float * w [[buffer(3)]],
                                  device const int * ids [[buffer(4)]],
                                  device float * y [[buffer(5)]],
                                  device atomic_uint * faults [[buffer(6)]],
                                  constant FoldArgs & args [[buffer(7)]],
                                  uint3 tgpig [[threadgroup_position_in_grid]],
                                  ushort tiisg [[thread_index_in_simdgroup]],
                                  ushort sgitg [[simdgroup_index_in_threadgroup]]);

template [[host_name("q6k_down_fold")]]
kernel void down_fold_kernel<Q6K>(device const block_q6_K * down [[buffer(0)]],
                                  device const block_q6_K * shared [[buffer(1)]],
                                  device const float * h [[buffer(2)]],
                                  device const float * w [[buffer(3)]],
                                  device const int * ids [[buffer(4)]],
                                  device float * y [[buffer(5)]],
                                  device atomic_uint * faults [[buffer(6)]],
                                  constant FoldArgs & args [[buffer(7)]],
                                  uint3 tgpig [[threadgroup_position_in_grid]],
                                  ushort tiisg [[thread_index_in_simdgroup]],
                                  ushort sgitg [[simdgroup_index_in_threadgroup]]);

template [[host_name("a4_down_fold")]]
kernel void down_fold_kernel<A4>(device const block_mlx_a4 * down [[buffer(0)]],
                                 device const block_mlx_a4 * shared [[buffer(1)]],
                                 device const float * h [[buffer(2)]],
                                 device const float * w [[buffer(3)]],
                                 device const int * ids [[buffer(4)]],
                                 device float * y [[buffer(5)]],
                                 device atomic_uint * faults [[buffer(6)]],
                                 constant FoldArgs & args [[buffer(7)]],
                                 uint3 tgpig [[threadgroup_position_in_grid]],
                                 ushort tiisg [[thread_index_in_simdgroup]],
                                 ushort sgitg [[simdgroup_index_in_threadgroup]]);

template [[host_name("a8_down_fold")]]
kernel void down_fold_kernel<A8>(device const block_mlx_a8 * down [[buffer(0)]],
                                 device const block_mlx_a8 * shared [[buffer(1)]],
                                 device const float * h [[buffer(2)]],
                                 device const float * w [[buffer(3)]],
                                 device const int * ids [[buffer(4)]],
                                 device float * y [[buffer(5)]],
                                 device atomic_uint * faults [[buffer(6)]],
                                 constant FoldArgs & args [[buffer(7)]],
                                 uint3 tgpig [[threadgroup_position_in_grid]],
                                 ushort tiisg [[thread_index_in_simdgroup]],
                                 ushort sgitg [[simdgroup_index_in_threadgroup]]);

template [[host_name("a4h_down_fold")]]
kernel void down_fold_kernel<A4H>(device const block_mlx_a4 * down [[buffer(0)]],
                                  device const block_mlx_a4 * shared [[buffer(1)]],
                                  device const float * h [[buffer(2)]],
                                  device const float * w [[buffer(3)]],
                                  device const int * ids [[buffer(4)]],
                                  device float * y [[buffer(5)]],
                                  device atomic_uint * faults [[buffer(6)]],
                                  constant FoldArgs & args [[buffer(7)]],
                                  uint3 tgpig [[threadgroup_position_in_grid]],
                                  ushort tiisg [[thread_index_in_simdgroup]],
                                  ushort sgitg [[simdgroup_index_in_threadgroup]]);

template [[host_name("a8h_down_fold")]]
kernel void down_fold_kernel<A8H>(device const block_mlx_a8 * down [[buffer(0)]],
                                  device const block_mlx_a8 * shared [[buffer(1)]],
                                  device const float * h [[buffer(2)]],
                                  device const float * w [[buffer(3)]],
                                  device const int * ids [[buffer(4)]],
                                  device float * y [[buffer(5)]],
                                  device atomic_uint * faults [[buffer(6)]],
                                  constant FoldArgs & args [[buffer(7)]],
                                  uint3 tgpig [[threadgroup_position_in_grid]],
                                  ushort tiisg [[thread_index_in_simdgroup]],
                                  ushort sgitg [[simdgroup_index_in_threadgroup]]);

template [[host_name("q8_0_down_fold")]]
kernel void down_fold_kernel<Q8_0>(device const block_q8_0 * down [[buffer(0)]],
                                   device const block_q8_0 * shared [[buffer(1)]],
                                   device const float * h [[buffer(2)]],
                                   device const float * w [[buffer(3)]],
                                   device const int * ids [[buffer(4)]],
                                   device float * y [[buffer(5)]],
                                   device atomic_uint * faults [[buffer(6)]],
                                   constant FoldArgs & args [[buffer(7)]],
                                   uint3 tgpig [[threadgroup_position_in_grid]],
                                   ushort tiisg [[thread_index_in_simdgroup]],
                                   ushort sgitg [[simdgroup_index_in_threadgroup]]);

template [[host_name("q5k_down_fold")]]
kernel void down_fold_kernel<Q5K>(device const block_q5_K * down [[buffer(0)]],
                                  device const block_q5_K * shared [[buffer(1)]],
                                  device const float * h [[buffer(2)]],
                                  device const float * w [[buffer(3)]],
                                  device const int * ids [[buffer(4)]],
                                  device float * y [[buffer(5)]],
                                  device atomic_uint * faults [[buffer(6)]],
                                  constant FoldArgs & args [[buffer(7)]],
                                  uint3 tgpig [[threadgroup_position_in_grid]],
                                  ushort tiisg [[thread_index_in_simdgroup]],
                                  ushort sgitg [[simdgroup_index_in_threadgroup]]);
