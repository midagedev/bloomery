// The Q4_K embedding: row `ids[t]` of a Q4_K table, dequantized into row t of
// an f32 output, one thread per value.
//
// An id at or past the table's rows is undefined input: the kernel raises the
// token-id site in `layer`, reads no table word, and writes NaN to every value
// of the row, which no later kernel can take for an embedding.
//
// The value is the dequantizer's (ggml's dequantize_row_q4_K): d1 * q - m1,
// where `d * sc`, `d * sc * q` and `dmin * m` are exact in f32, so a fused or
// an unfused subtraction gives the same bits.
//
// This file comes after matvec_q4k.metal (QK_K, block_q4_K) and fault.metal.

#include <metal_stdlib>

using namespace metal;

#pragma METAL fp contract(off)

// The embedding's argument block, bound by value at slot 3.
struct EmbedArgs {
    uint vocab;    // the table's rows
    uint nblocks;  // the table's columns / 256
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
