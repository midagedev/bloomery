// The decode owners of the 32-value block types with a fifth code bit, Q5_0
// and Q5_1: the block layouts and the value rule each mat-vec and embedding
// instantiates, in the shape quant.metal's owners have (its header lists what
// an owner holds; this file follows it).
//
// A block holds 32 values: sixteen low nibbles, sixteen high nibbles of the
// same bytes, and the fifth bit of each of the 32 values in one 32-bit word,
// value `j` taking bit `j`. Q5_0 is `d·(q − 16)`, Q5_1 is `d·q + m`.
//
// The lane map is llama.cpp's kernel_mul_mv_q5_0_f32 and kernel_mul_mv_q5_1_f32
// (the shared q4_0/q4_1 shape; MIT licence, see THIRD_PARTY_NOTICES.md):
//   ggml/src/ggml-metal/kernels/mul_mv.metal
// Two lanes split a block, each taking eight bytes: the lane's low-nibble
// values are `il ... il + 7` and its high-nibble values `16 + il ... 16 + il
// + 7`, so the 32 lanes cover sixteen blocks, 512 values, a pass. The
// activations of the high nibbles and of the odd bytes are loaded divided by
// the power of two their mask leaves in the code word (1/256, 1/16, 1/4096),
// the uint16 nibble trick of quant.metal's Q4_K, and the fifth bit enters
// as an `or` into the masked word. The sum associations (the pair sums, the
// two accumulators, `d·(acc0 + acc1) + sumy·m`) are llama.cpp's.
//
// `block_q5_0` is 22 bytes and `block_q5_1` 24: a block starts 2-byte aligned
// and no further, so every read of a block is of 16-bit words (the fifth-bit
// word as two of them), never of a 32-bit one.
//
// This file comes after quant.metal (FOR_UNROLL and LoadMode) and before
// gemm_shared.metal, so the tensor-op GEMMs' source holds it too.

#include <metal_stdlib>

using namespace metal;

#pragma METAL fp contract(off)

struct block_q5_0 {
    half    d;
    uint8_t qh[4];
    uint8_t qs[16];
};

static_assert(sizeof(block_q5_0) == 22, "block_q5_0 is 22 bytes");

struct block_q5_1 {
    half    d;
    half    m;
    uint8_t qh[4];
    uint8_t qs[16];
};

static_assert(sizeof(block_q5_1) == 24, "block_q5_1 is 24 bytes");

// What Q5_0 and Q5_1 share: the lane map and its registers, the load, the
// code of a value and the dot of the codes. Each owner delegates to it and adds
// its block layout's word offsets and the scale (and, for Q5_1, the min) it
// combines them with. (Metal has no derived classes, so the delegation is
// quant.metal's own, Q5K's onto Q4K's.)
struct Q5Bits {
    // The lane's map and registers: ix picks the lane's blocks (sixteen
    // apart), il its eight bytes of the block (0 or 8). yl holds the loaded
    // activations scaled for the masks, sumy their two sums.
    struct Lane {
        short   ix;
        short   il;
        float   yl[16];
        float   sumy[2];
    };

    static inline Lane lane_of(ushort tiisg) {
        Lane lane;
        lane.ix = (short)(tiisg / 2);        // 0...15
        lane.il = (short)((tiisg % 2) * 8);  // 0 or 8
        return lane;
    }

    static inline int first_block(const thread Lane &lane) {
        return lane.ix;
    }

    static inline int block_stride() {
        return 16;
    }

    static inline int values_per_block() {
        return 32;
    }

    // The lane's low-nibble values' offset in a block; its high-nibble
    // values are sixteen further.
    static inline int input_offset(const thread Lane &lane) {
        return lane.il;
    }

    // The activations of one block, in the lane's registers. `gain` and
    // `sum_sq` are read by the norm load only, `scale` by the route load.
    template <LoadMode LOAD>
    static inline void load(thread Lane &lane,
                            device const float * block,
                            device const float * gain,
                            float scale,
                            thread float * sum_sq) {
        const int base = input_offset(lane);
        lane.sumy[0] = 0.f;
        lane.sumy[1] = 0.f;
        FOR_UNROLL (short i = 0; i < 8; i += 2) {
            float a0 = block[base + i + 0];
            float a1 = block[base + i + 1];
            float a2 = block[base + i + 16];
            float a3 = block[base + i + 17];
            if (LOAD == LOAD_ROUTE) {
                a0 *= scale;
                a1 *= scale;
                a2 *= scale;
                a3 *= scale;
            } else if (LOAD == LOAD_NORM) {
                *sum_sq += (a0 * a0 + a1 * a1) + (a2 * a2 + a3 * a3);
                a0 *= gain[base + i + 0];
                a1 *= gain[base + i + 1];
                a2 *= gain[base + i + 16];
                a3 *= gain[base + i + 17];
            }
            lane.sumy[0] += a0 + a1;
            lane.yl[i + 0] = a0;
            lane.yl[i + 1] = a1 / 256.f;
            lane.sumy[1] += a2 + a3;
            lane.yl[i + 8] = a2 / 16.f;
            lane.yl[i + 9] = a3 / 4096.f;
        }
    }

    // The fifth-bit word of a block, read as the two 16-bit words it is
    // stored in.
    static inline uint bits_of(device const uint16_t * words) {
        return (uint)words[0] | ((uint)words[1] << 16);
    }

    // The five-bit code of value `j` of a block: the nibble of its byte (low
    // for the first sixteen values, high for the rest) and its bit of `qh`.
    static inline uint code_of(device const uint8_t * qs, uint qh, uint j) {
        const uint nibble = j < 16 ? (qs[j] & 0x0Fu) : (qs[j - 16] >> 4);
        return nibble | (((qh >> j) & 1u) << 4);
    }

    // The lane's eight bytes of codes dotted with its loaded activations:
    // acc[0] over the low nibbles, acc[1] over the high nibbles. Each product
    // is the activation (scaled down by the mask's power of two) times the
    // masked word with the fifth bit or-ed in at the mask's own place.
    static inline float2 dots(device const uint16_t * qs,
                              uint qh,
                              const thread Lane & lane) {
        const short il = lane.il;
        float2 acc = float2(0.f, 0.f);
        FOR_UNROLL (short i = 0; i < 8; i += 2) {
            acc[0] += lane.yl[i + 0] * (float)((qs[i / 2] & 0x000F) | ((qh >> (i + 0 + il) << 4) & 0x00010))
                    + lane.yl[i + 1] * (float)((qs[i / 2] & 0x0F00) | ((qh >> (i + 1 + il) << 12) & 0x01000));
            acc[1] += lane.yl[i + 8] * (float)((qs[i / 2] & 0x00F0) | ((qh >> (i + 0 + il + 16) << 8) & 0x00100))
                    + lane.yl[i + 9] * (float)((qs[i / 2] & 0xF000) | ((qh >> (i + 1 + il + 16) << 16) & 0x10000));
        }
        return acc;
    }
};

// GGUF's Q5_0 decode owner: `y = d·(q − 16)`, q the five-bit code.
struct Q5_0 {
    typedef block_q5_0 Block;
    typedef Q5Bits::Lane Lane;

    static inline Lane lane_of(ushort tiisg) {
        return Q5Bits::lane_of(tiisg);
    }

    static inline int first_block(const thread Lane &lane) {
        return Q5Bits::first_block(lane);
    }

    static inline int block_stride() {
        return Q5Bits::block_stride();
    }

    static inline int values_per_block() {
        return Q5Bits::values_per_block();
    }

    static inline int input_offset(const thread Lane &lane) {
        return Q5Bits::input_offset(lane);
    }

    template <LoadMode LOAD>
    static inline void load(thread Lane &lane,
                            device const float * block,
                            device const float * gain,
                            float scale,
                            thread float * sum_sq) {
        Q5Bits::load<LOAD>(lane, block, gain, scale, sum_sq);
    }

    // One block's contribution: the code dot less 16 times the activation
    // sum, then the scale once.
    static inline float partial(device const Block & blk,
                                const thread Lane & lane) {
        device const uint16_t * words = (device const uint16_t *)&blk;
        const float sumy = lane.sumy[0] + lane.sumy[1];
        const float2 acc = Q5Bits::dots(words + 3 + lane.il / 2, Q5Bits::bits_of(words + 1), lane);
        return (float)blk.d * (sumy * -16.f + acc[0] + acc[1]);
    }

    // Value `j` of a block, the host reader's decode: the code less 16
    // times the scale, exact in f32 (a five-bit integer times eleven bits).
    static inline float value(device const Block & blk, uint j) {
        const uint code = Q5Bits::code_of(blk.qs, Q5Bits::bits_of((device const uint16_t *)&blk + 1), j);
        return (float)((int)code - 16) * (float)blk.d;
    }

    // Sixteen consecutive values of a block, [16*il, 16*il + 16) in natural
    // order (value 16*il + j at reg[j/4][j%4]), the GEMM's A staging: the
    // owner's own `value`, rounded once into the operand type.
    template <typename T>
    static inline void dequant16(device const Block & blk,
                                 short il,
                                 thread matrix<T, 4, 4> & reg) {
        FOR_UNROLL (short j = 0; j < 16; ++j) {
            reg[j / 4][j % 4] = value(blk, (uint)(16 * il + j));
        }
    }
};

// GGUF's Q5_1 decode owner: `y = d·q + m`, q the five-bit code.
struct Q5_1 {
    typedef block_q5_1 Block;
    typedef Q5Bits::Lane Lane;

    static inline Lane lane_of(ushort tiisg) {
        return Q5Bits::lane_of(tiisg);
    }

    static inline int first_block(const thread Lane &lane) {
        return Q5Bits::first_block(lane);
    }

    static inline int block_stride() {
        return Q5Bits::block_stride();
    }

    static inline int values_per_block() {
        return Q5Bits::values_per_block();
    }

    static inline int input_offset(const thread Lane &lane) {
        return Q5Bits::input_offset(lane);
    }

    template <LoadMode LOAD>
    static inline void load(thread Lane &lane,
                            device const float * block,
                            device const float * gain,
                            float scale,
                            thread float * sum_sq) {
        Q5Bits::load<LOAD>(lane, block, gain, scale, sum_sq);
    }

    // One block's contribution: the code dot under the scale, plus the
    // activation sum under the min.
    static inline float partial(device const Block & blk,
                                const thread Lane & lane) {
        device const uint16_t * words = (device const uint16_t *)&blk;
        const float sumy = lane.sumy[0] + lane.sumy[1];
        const float2 acc = Q5Bits::dots(words + 4 + lane.il / 2, Q5Bits::bits_of(words + 2), lane);
        return (float)blk.d * (acc[0] + acc[1]) + sumy * (float)blk.m;
    }

    // Value `j` of a block, the host reader's decode: the product exact in
    // f32, the sum rounded once, as the reader's fused multiply-add rounds.
    static inline float value(device const Block & blk, uint j) {
        const uint code = Q5Bits::code_of(blk.qs, Q5Bits::bits_of((device const uint16_t *)&blk + 2), j);
        return (float)code * (float)blk.d + (float)blk.m;
    }

    template <typename T>
    static inline void dequant16(device const Block & blk,
                                 short il,
                                 thread matrix<T, 4, 4> & reg) {
        FOR_UNROLL (short j = 0; j < 16; ++j) {
            reg[j / 4][j % 4] = value(blk, (uint)(16 * il + j));
        }
    }
};
