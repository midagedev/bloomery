// The routed expert gate-up pair and the folded down. One dispatch each:
//
//   q4k_pair_swiglu   y[slot, r] = silu(gate_r · x̂) * (up_r · x̂),
//                     x̂ = x * gain / sqrt(mean(x^2) + eps)
//   q6k_pair_swiglu   the same over block_q6_K
//   q4k_down_fold     y[r] += sum over slots of w_s * (down_s[r] · h_s)
//   q6k_down_fold     the same over block_q6_K
//
// Adapted from llama.cpp's kernel_mul_mv_q4_K_f32 and kernel_mul_mv_q6_K_f32
// (MIT licence; see THIRD_PARTY_NOTICES.md):
//   ggml/src/ggml-metal/kernels/mul_mv.metal
// The lane map, the block loop, the nibble and 6-bit unpacking and the
// simd_sum reduction are llama.cpp's, the same ones the dense mat-vec uses.
//
// The pair tile is one output row per SIMD group and two SIMD groups in the
// threadgroup. The group's two weight rows are the gate row and the up row
// of that one output, which is the dense mat-vec's register shape with the
// two output rows replaced by the pair. grid.y is the slot. Slot 8 reads the
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
// The fold's K' is nine slots of K 512, eighteen blocks. A lane owns blocks
// by the dense stride across that whole K', not slot by slot: Q4_K steps four
// blocks (a lane owns four or five), Q6_K steps two (a lane owns nine). The
// route weight of the block's slot scales the activation as it is loaded, and
// the residual add is the store. There is no second dispatch.
//
// An id past the stack raises ExpertId and writes NaN before any weight is
// loaded. The pair writes NaN only in the bad slot's row; the fold writes NaN
// in every row of the threadgroup, because every row reads every slot.
//
// This file comes after matvec_q6k.metal (QK_K, FOR_UNROLL, the block
// structs), norm.metal (rms_scale, rms_usable) and fault.metal (raise).

#include <metal_stdlib>

using namespace metal;

#pragma METAL fp contract(off)

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
// the select, so both quants and both tensors follow it.
inline device const char * expert_base(bool shared_slot,
                                       device const char * shexp,
                                       device const char * stack) {
    return shared_slot ? shexp : stack;
}

// silu(gate) * up. No route weight: the fold multiplies by it.
inline float pair_swiglu(float gate, float up) {
    return gate / (1.0f + exp(-gate)) * up;
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

// A slot's row the pair cannot compute: `site` raised in the layer, NaN in
// the row.
inline void pair_write_bad(device float * y, uint slot, uint stride, uint row,
                           device atomic_uint * faults, uint layer, uint site) {
    raise(faults, layer, site);
    y[(ulong)slot * stride + row] = NAN;
}

inline void fold_write_bad(device float * y, int row0, int rows,
                           device atomic_uint * faults, uint layer) {
    raise(faults, layer, BLOOMERY_FAULT_SITE_EXPERT_ID);
    if (row0 < rows) {
        y[(uint)row0] = NAN;
    }
    if (row0 + 1 < rows) {
        y[(uint)row0 + 1u] = NAN;
    }
}

// One Q4_K block's contribution. The subtraction keeps the dense mat-vec's
// association: d times the nibble sum, minus dmin times the activation sum.
inline float q4k_partial(device const block_q4_K & blk,
                         thread const float * yl,
                         thread const float * yh,
                         float4 sumy,
                         short iq,
                         short ir) {
    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;
    device const uint16_t * sc = (device const uint16_t *)blk.scales + iq;
    device const uint16_t * q1 = (device const uint16_t *)blk.qs + 16 * iq + 4 * ir;
    device const half * dh = &blk.d;

    sc16[0] = sc[0] & kmask1;
    sc16[1] = sc[2] & kmask1;
    sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
    sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

    device const uint16_t * q2 = q1 + 32;
    float4 acc1 = {0.f, 0.f, 0.f, 0.f};
    float4 acc2 = {0.f, 0.f, 0.f, 0.f};
    FOR_UNROLL (short i = 0; i < 4; ++i) {
        acc1[0] += yl[2 * i + 0] * (q1[i] & 0x000F);
        acc1[1] += yl[2 * i + 1] * (q1[i] & 0x0F00);
        acc1[2] += yl[2 * i + 8] * (q1[i] & 0x00F0);
        acc1[3] += yl[2 * i + 9] * (q1[i] & 0xF000);
        acc2[0] += yh[2 * i + 0] * (q2[i] & 0x000F);
        acc2[1] += yh[2 * i + 1] * (q2[i] & 0x0F00);
        acc2[2] += yh[2 * i + 8] * (q2[i] & 0x00F0);
        acc2[3] += yh[2 * i + 9] * (q2[i] & 0xF000);
    }

    return dh[0] * ((acc1[0] + 1.f/256.f * acc1[1]) * sc8[0] +
                    (acc1[2] + 1.f/256.f * acc1[3]) * sc8[1] * 1.f/16.f +
                    (acc2[0] + 1.f/256.f * acc2[1]) * sc8[4] +
                    (acc2[2] + 1.f/256.f * acc2[3]) * sc8[5] * 1.f/16.f)
         - dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                    sumy[2] * sc8[6] + sumy[3] * sc8[7]);
}

// One Q6_K block's contribution. The high bits of each 6-bit value are split
// out before the offset, so the offset stays on the assembled value.
inline float q6k_partial(device const block_q6_K & blk,
                         thread const float * yl,
                         short ip,
                         short l0) {
    constexpr uint8_t kmask1 = 0x03;
    constexpr uint8_t kmask2 = 0x0C;
    constexpr uint8_t kmask3 = 0x30;
    constexpr uint8_t kmask4 = 0xC0;
    const short q_offset_l = 64 * ip + l0;
    const short q_offset_h = 32 * ip + l0;
    const short is = 8 * ip + l0 / 16;

    device const uint8_t * q1 = blk.ql + q_offset_l;
    device const uint8_t * q2 = q1 + 32;
    device const uint8_t * qh = blk.qh + q_offset_h;
    device const char * sc = blk.scales + is;
    device const half * dh = &blk.d;
    float4 sums = {0.f, 0.f, 0.f, 0.f};
    FOR_UNROLL (short l = 0; l < 4; ++l) {
        uint8_t h0 = (uint8_t)((qh[l] & kmask1) << 4);
        uint8_t h1 = (uint8_t)((qh[l] & kmask2) << 2);
        uint8_t h2 = (uint8_t)((qh[l] & kmask3) << 0);
        uint8_t h3 = (uint8_t)((qh[l] & kmask4) >> 2);
        sums[0] += yl[4 * l + 0] * ((char)((q1[l] & 0xF) | h0) - 32);
        sums[1] += yl[4 * l + 1] * ((char)((q2[l] & 0xF) | h1) - 32);
        sums[2] += yl[4 * l + 2] * ((char)((q1[l] >> 4) | h2) - 32);
        sums[3] += yl[4 * l + 3] * ((char)((q2[l] >> 4) | h3) - 32);
    }
    return dh[0] * (sums[0] * sc[0] + sums[1] * sc[2] + sums[2] * sc[4] + sums[3] * sc[6]);
}

// What a load does to each value it loads.
enum ExpertLoad {
    // The fold's: the route weight of the block's slot multiplies it.
    LOAD_ROUTE,
    // The pair's: its square joins the lane's sum of squares, then its gain
    // multiplies it (the mat-vec's norm on load, in its order).
    LOAD_NORM,
};

// The activations of one Q4_K block, in the dense lane's registers. `gain`
// (the block's gain) and `sum_sq` (the lane's sum of squares) are the pair's;
// the fold passes null for both, and LOAD_ROUTE reads neither.
template <ExpertLoad LOAD>
inline void q4k_load(device const float * block, device const float * gain, short iq, short ir,
                     float scale, thread float * yl, thread float * yh, thread float4 & sumy,
                     thread float * sum_sq) {
    const int base = 64 * iq + 8 * ir;
    sumy = float4(0.f, 0.f, 0.f, 0.f);
    for (short i = 0; i < 8; ++i) {
        float a0 = block[base + i + 0];
        float a1 = block[base + i + 32];
        float a2 = block[base + i + 128];
        float a3 = block[base + i + 160];
        if (LOAD == LOAD_ROUTE) {
            a0 *= scale;
            a1 *= scale;
            a2 *= scale;
            a3 *= scale;
        } else {
            *sum_sq += (a0 * a0 + a1 * a1) + (a2 * a2 + a3 * a3);
            a0 *= gain[base + i + 0];
            a1 *= gain[base + i + 32];
            a2 *= gain[base + i + 128];
            a3 *= gain[base + i + 160];
        }
        yl[i + 0] = a0;
        sumy[0] += a0;
        yl[i + 8] = a1;
        sumy[1] += a1;
        yh[i + 0] = a2;
        sumy[2] += a2;
        yh[i + 8] = a3;
        sumy[3] += a3;
    }
}

template <ExpertLoad LOAD>
inline void q6k_load(device const float * block, device const float * gain, short ip, short l0,
                     float scale, thread float * yl, thread float * sum_sq) {
    const int base = 128 * ip + l0;
    for (short l = 0; l < 4; ++l) {
        float a0 = block[base + l + 0];
        float a1 = block[base + l + 32];
        float a2 = block[base + l + 64];
        float a3 = block[base + l + 96];
        if (LOAD == LOAD_ROUTE) {
            a0 *= scale;
            a1 *= scale;
            a2 *= scale;
            a3 *= scale;
        } else {
            *sum_sq += (a0 * a0 + a1 * a1) + (a2 * a2 + a3 * a3);
            a0 *= gain[base + l + 0];
            a1 *= gain[base + l + 32];
            a2 *= gain[base + l + 64];
            a3 *= gain[base + l + 96];
        }
        yl[4 * l + 0] = a0;
        yl[4 * l + 1] = a1;
        yl[4 * l + 2] = a2;
        yl[4 * l + 3] = a3;
    }
}

// The rows of one expert. `expert` is zero for the shared slot, and the base
// is then the shared tensor, so the byte offset stays zero.
template <typename Block>
inline device const Block * pair_rows(device const Block * stack,
                                      device const Block * shared,
                                      bool shared_slot,
                                      int expert,
                                      uint rows,
                                      uint nblocks,
                                      int row) {
    device const char * base = expert_base(shared_slot,
                                           (device const char *)shared,
                                           (device const char *)stack);
    ulong bytes = (ulong)expert * (ulong)rows * (ulong)nblocks * sizeof(Block);
    return ((device const Block *)(base + bytes)) + (ulong)row * (ulong)nblocks;
}

// One block of the folded K'. Slot s owns blocks 2s and 2s+1; the shared slot
// is the shared tensor. The stack is not contiguous across experts, so the
// pointer is this block's own.
template <typename Block>
inline device const Block * fold_block(device const Block * stack,
                                       device const Block * shared,
                                       device const int * ids,
                                       uint rows,
                                       int row,
                                       int ib) {
    int slot = ib >> 1;
    int local = ib & 1;
    bool shared_slot = slot == BLOOMERY_EXPERT_SHARED;
    int expert = shared_slot ? 0 : ids[slot];
    device const char * base = expert_base(shared_slot,
                                           (device const char *)shared,
                                           (device const char *)stack);
    ulong index = ((ulong)expert * (ulong)rows + (ulong)row) * 2ul + (ulong)local;
    return ((device const Block *)base) + index;
}

kernel void q4k_pair_swiglu(device const block_q4_K * gate [[buffer(0)]],
                            device const block_q4_K * up [[buffer(1)]],
                            device const block_q4_K * shexp_gate [[buffer(2)]],
                            device const block_q4_K * shexp_up [[buffer(3)]],
                            device const float * x [[buffer(4)]],
                            device const float * gain [[buffer(5)]],
                            device const int * ids [[buffer(6)]],
                            device float * y [[buffer(7)]],
                            device atomic_uint * faults [[buffer(8)]],
                            constant PairArgs & args [[buffer(9)]],
                            uint3 tgpig [[threadgroup_position_in_grid]],
                            ushort tiisg [[thread_index_in_simdgroup]],
                            ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;
    const uint slot = tgpig.y;
    if (slot >= BLOOMERY_EXPERT_SLOTS) {
        return;
    }
    // One output row per SIMD group, two SIMD groups in the threadgroup.
    const int row0 = (int)(tgpig.x * 2u + (uint)sgitg);
    if (row0 >= (int)args.rows) {
        return;
    }

    int expert = 0;
    if (pair_expert(ids, slot, args.n_experts, expert)) {
        if (tiisg == 0u) {
            pair_write_bad(y, slot, args.stride, (uint)row0, faults, args.layer,
                           BLOOMERY_FAULT_SITE_EXPERT_ID);
        }
        return;
    }

    device const block_q4_K * grow =
        pair_rows(gate, shexp_gate, slot == BLOOMERY_EXPERT_SHARED, expert, args.rows, args.nblocks, row0);
    device const block_q4_K * urow =
        pair_rows(up, shexp_up, slot == BLOOMERY_EXPERT_SHARED, expert, args.rows, args.nblocks, row0);
    float sumg = 0.f;
    float sumu = 0.f;
    float sum_sq = 0.f;
    for (int ib = ix; ib < (int)args.nblocks; ib += 4) {
        float yl[16];
        float yh[16];
        float4 sumy;
        q4k_load<LOAD_NORM>(x + (ulong)ib * QK_K, gain + (ulong)ib * QK_K, iq, ir, 1.f, yl, yh,
                            sumy, &sum_sq);
        sumg += q4k_partial(grow[ib], yl, yh, sumy, iq, ir);
        sumu += q4k_partial(urow[ib], yl, yh, sumy, iq, ir);
    }
    const float sum_squares = simd_sum(sum_sq);
    const float scale = rms_scale(sum_squares, args.nblocks * QK_K, args.eps);
    if (!rms_usable(sum_squares, scale)) {
        if (tiisg == 0u) {
            pair_write_bad(y, slot, args.stride, (uint)row0, faults, args.layer,
                           BLOOMERY_FAULT_SITE_NORM_QUANT);
        }
        return;
    }
    float g = simd_sum(sumg) * scale;
    float u = simd_sum(sumu) * scale;
    if (tiisg == 0u) {
        y[(ulong)slot * args.stride + (uint)row0] = pair_swiglu(g, u);
    }
}

kernel void q6k_pair_swiglu(device const block_q6_K * gate [[buffer(0)]],
                            device const block_q6_K * up [[buffer(1)]],
                            device const block_q6_K * shexp_gate [[buffer(2)]],
                            device const block_q6_K * shexp_up [[buffer(3)]],
                            device const float * x [[buffer(4)]],
                            device const float * gain [[buffer(5)]],
                            device const int * ids [[buffer(6)]],
                            device float * y [[buffer(7)]],
                            device atomic_uint * faults [[buffer(8)]],
                            constant PairArgs & args [[buffer(9)]],
                            uint3 tgpig [[threadgroup_position_in_grid]],
                            ushort tiisg [[thread_index_in_simdgroup]],
                            ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const short tid = tiisg / 2;
    const short ix = tiisg % 2;
    const short ip = tid / 8;
    const short il = tid % 8;
    const short l0 = 4 * il;
    const uint slot = tgpig.y;
    if (slot >= BLOOMERY_EXPERT_SLOTS) {
        return;
    }
    const int row0 = (int)(tgpig.x * 2u + (uint)sgitg);
    if (row0 >= (int)args.rows) {
        return;
    }

    int expert = 0;
    if (pair_expert(ids, slot, args.n_experts, expert)) {
        if (tiisg == 0u) {
            pair_write_bad(y, slot, args.stride, (uint)row0, faults, args.layer,
                           BLOOMERY_FAULT_SITE_EXPERT_ID);
        }
        return;
    }

    device const block_q6_K * grow =
        pair_rows(gate, shexp_gate, slot == BLOOMERY_EXPERT_SHARED, expert, args.rows, args.nblocks, row0);
    device const block_q6_K * urow =
        pair_rows(up, shexp_up, slot == BLOOMERY_EXPERT_SHARED, expert, args.rows, args.nblocks, row0);
    float sumg = 0.f;
    float sumu = 0.f;
    float sum_sq = 0.f;
    for (int ib = ix; ib < (int)args.nblocks; ib += 2) {
        float yl[16];
        q6k_load<LOAD_NORM>(x + (ulong)ib * QK_K, gain + (ulong)ib * QK_K, ip, l0, 1.f, yl,
                            &sum_sq);
        sumg += q6k_partial(grow[ib], yl, ip, l0);
        sumu += q6k_partial(urow[ib], yl, ip, l0);
    }
    const float sum_squares = simd_sum(sum_sq);
    const float scale = rms_scale(sum_squares, args.nblocks * QK_K, args.eps);
    if (!rms_usable(sum_squares, scale)) {
        if (tiisg == 0u) {
            pair_write_bad(y, slot, args.stride, (uint)row0, faults, args.layer,
                           BLOOMERY_FAULT_SITE_NORM_QUANT);
        }
        return;
    }
    float g = simd_sum(sumg) * scale;
    float u = simd_sum(sumu) * scale;
    if (tiisg == 0u) {
        y[(ulong)slot * args.stride + (uint)row0] = pair_swiglu(g, u);
    }
}

kernel void q4k_down_fold(device const block_q4_K * down [[buffer(0)]],
                          device const block_q4_K * shared [[buffer(1)]],
                          device const float * h [[buffer(2)]],
                          device const float * w [[buffer(3)]],
                          device const int * ids [[buffer(4)]],
                          device float * y [[buffer(5)]],
                          device atomic_uint * faults [[buffer(6)]],
                          constant FoldArgs & args [[buffer(7)]],
                          uint3 tgpig [[threadgroup_position_in_grid]],
                          ushort tiisg [[thread_index_in_simdgroup]],
                          ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;
    // Two rows per SIMD group, two SIMD groups: four rows in the threadgroup.
    const int row0 = (int)((tgpig.x * 2u + (uint)sgitg) * 2u);
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

    float sum0 = 0.f;
    float sum1 = 0.f;
    const int row1 = row0 + 1;
    const bool has1 = row1 < (int)args.rows;
    for (int ib = ix; ib < BLOOMERY_EXPERT_BLOCKS; ib += 4) {
        float scale = w[ib >> 1];
        float yl[16];
        float yh[16];
        float4 sumy;
        q4k_load<LOAD_ROUTE>(h + (ulong)ib * QK_K, nullptr, iq, ir, scale, yl, yh, sumy, nullptr);
        device const block_q4_K * b0 =
            fold_block<block_q4_K>(down, shared, ids, args.rows, row0, ib);
        sum0 += q4k_partial(*b0, yl, yh, sumy, iq, ir);
        if (has1) {
            device const block_q4_K * b1 =
                fold_block<block_q4_K>(down, shared, ids, args.rows, row1, ib);
            sum1 += q4k_partial(*b1, yl, yh, sumy, iq, ir);
        }
    }
    float d0 = simd_sum(sum0);
    float d1 = simd_sum(sum1);
    if (tiisg == 0u) {
        y[(uint)row0] += d0;
        if (has1) {
            y[(uint)row1] += d1;
        }
    }
}

kernel void q6k_down_fold(device const block_q6_K * down [[buffer(0)]],
                          device const block_q6_K * shared [[buffer(1)]],
                          device const float * h [[buffer(2)]],
                          device const float * w [[buffer(3)]],
                          device const int * ids [[buffer(4)]],
                          device float * y [[buffer(5)]],
                          device atomic_uint * faults [[buffer(6)]],
                          constant FoldArgs & args [[buffer(7)]],
                          uint3 tgpig [[threadgroup_position_in_grid]],
                          ushort tiisg [[thread_index_in_simdgroup]],
                          ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const short tid = tiisg / 2;
    const short ix = tiisg % 2;
    const short ip = tid / 8;
    const short il = tid % 8;
    const short l0 = 4 * il;
    const int row0 = (int)((tgpig.x * 2u + (uint)sgitg) * 2u);
    if (row0 >= (int)args.rows) {
        return;
    }

    if (!fold_ids_ok(ids, args.n_experts)) {
        if (tiisg == 0u) {
            fold_write_bad(y, row0, (int)args.rows, faults, args.layer);
        }
        return;
    }

    float sum0 = 0.f;
    float sum1 = 0.f;
    const int row1 = row0 + 1;
    const bool has1 = row1 < (int)args.rows;
    for (int ib = ix; ib < BLOOMERY_EXPERT_BLOCKS; ib += 2) {
        float scale = w[ib >> 1];
        float yl[16];
        q6k_load<LOAD_ROUTE>(h + (ulong)ib * QK_K, nullptr, ip, l0, scale, yl, nullptr);
        device const block_q6_K * b0 =
            fold_block<block_q6_K>(down, shared, ids, args.rows, row0, ib);
        sum0 += q6k_partial(*b0, yl, ip, l0);
        if (has1) {
            device const block_q6_K * b1 =
                fold_block<block_q6_K>(down, shared, ids, args.rows, row1, ib);
            sum1 += q6k_partial(*b1, yl, ip, l0);
        }
    }
    float d0 = simd_sum(sum0);
    float d1 = simd_sum(sum1);
    if (tiisg == 0u) {
        y[(uint)row0] += d0;
        if (has1) {
            y[(uint)row1] += d1;
        }
    }
}
