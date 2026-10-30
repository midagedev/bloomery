// The embedding kernels: row `ids[t]` of a quantized table, dequantized into
// row t of an f32 output, one thread per value — `embed_q4k` over a Q4_K
// table, and one body over the owners' value decode (quant.metal, not a
// second copy of the rule) for the others: `embed_a4` over MLX's affine
// 4-bit groups, `embed_a8` over their 8-bit form, `embed_q8_0` over Q8_0,
// `embed_q5k` over Q5_K.
//
// An id at or past the table's rows is undefined input: the kernel raises the
// token-id site in `layer`, reads no table word, and writes NaN to every value
// of the row, which no later kernel can take for an embedding.
//
// The Q4_K value is the dequantizer's (ggml's dequantize_row_q4_K): d1 * q -
// m1, where `d * sc`, `d * sc * q` and `dmin * m` are exact in f32, so a fused
// or an unfused subtraction gives the same bits. The others' value is the
// owner's: A4's and A8's the host reader's `s·q + b` with the product exact
// and the sum rounded once (the fp-contract pragma holds the order), Q8_0's
// `q·d` exact, Q5_K's `d1·q − m1` with the subtraction the one rounding — so
// the embed's bits are the reader's.
//
// This file comes after quant.metal (QK_K, the block structs, the owners)
// and fault.metal (raise).

#include <metal_stdlib>

using namespace metal;

#pragma METAL fp contract(off)

// The embedding's argument block, bound by value at slot 3.
struct EmbedArgs {
    uint vocab;    // the table's rows
    uint nblocks;  // the table's columns / the type's block values
    uint layer;    // the layer a raise is recorded in
    uint count;    // ids to embed
};

static_assert(sizeof(EmbedArgs) == 16, "EmbedArgs is 16 bytes: the host packs four words");

// ggml's get_scale_min_k4: the 6-bit scale and min of sub-block j.
inline uint2 scale_min_k4(int j, device const uint8_t * q)
{
    if (j < 4) {
        return uint2(q[j] & 63, q[j + 4] & 63);
    }
    return uint2((q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4),
                 (q[j + 4] >> 4) | ((q[j] >> 6) << 4));
}

kernel void embed_q4k(device const block_q4_K * table [[buffer(0)]],
                      device const uint       * ids [[buffer(1)]],
                      device float            * out [[buffer(2)]],
                      constant EmbedArgs      & args [[buffer(3)]],
                      device atomic_uint      * faults [[buffer(4)]],
                      uint gid [[thread_position_in_grid]])
{
    const uint width = args.nblocks * QK_K;
    if ((ulong)gid >= (ulong)args.count * (ulong)width) {
        return;
    }
    const uint t = gid / width;
    const uint k = gid % width;
    const uint id = ids[t];
    float value;
    if (id < args.vocab) {
        device const block_q4_K & b = table[(ulong)id * args.nblocks + k / QK_K];
        const uint j = k % QK_K;          // the value's place in the block
        const uint group = j / 64;        // one of four 64-value groups
        const uint high = (j % 64) / 32;  // the second half of a group takes the high nibbles
        const uint lane = j % 32;
        const uint2 sm = scale_min_k4(2 * group + high, b.scales);
        const float d1 = float(b.d) * float(sm.x);
        const float m1 = float(b.dmin) * float(sm.y);
        const uint q = b.qs[32 * group + lane];
        value = float(high != 0 ? (q >> 4) : (q & 0x0F)) * d1 - m1;
    } else {
        if (k == 0) {
            raise(faults, args.layer, BLOOMERY_FAULT_SITE_TOKEN_ID);
        }
        value = NAN;
    }
    out[gid] = value;
}

// A table over a decode owner: the owner's value of each place.
template <typename Q>
kernel void embed_kernel(device const typename Q::Block * table [[buffer(0)]],
                         device const uint       * ids [[buffer(1)]],
                         device float            * out [[buffer(2)]],
                         constant EmbedArgs      & args [[buffer(3)]],
                         device atomic_uint      * faults [[buffer(4)]],
                         uint gid [[thread_position_in_grid]])
{
    // The args' block count counts the owner's blocks.
    const uint per = (uint)Q::values_per_block();
    const uint width = args.nblocks * per;
    if ((ulong)gid >= (ulong)args.count * (ulong)width) {
        return;
    }
    const uint t = gid / width;
    const uint k = gid % width;
    const uint id = ids[t];
    float v;
    if (id < args.vocab) {
        device const typename Q::Block & b = table[(ulong)id * args.nblocks + k / per];
        v = Q::value(b, k % per);
    } else {
        if (k == 0) {
            raise(faults, args.layer, BLOOMERY_FAULT_SITE_TOKEN_ID);
        }
        v = NAN;
    }
    out[gid] = v;
}

#define EMBED_ENTRY(NAME, Q)                                                      \
    template [[host_name(#NAME)]] kernel void embed_kernel<Q>(                    \
        device const Q::Block * table [[buffer(0)]],                              \
        device const uint     * ids [[buffer(1)]],                                \
        device float          * out [[buffer(2)]],                                \
        constant EmbedArgs    & args [[buffer(3)]],                               \
        device atomic_uint    * faults [[buffer(4)]],                             \
        uint gid [[thread_position_in_grid]]);

EMBED_ENTRY(embed_a4, A4)
EMBED_ENTRY(embed_a8, A8)
EMBED_ENTRY(embed_q8_0, Q8_0)
EMBED_ENTRY(embed_q5k, Q5K)
