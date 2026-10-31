// The decode owner of each quant: the one place a quantized weight block is
// unpacked. A quant owns
//
//   Block        its GGUF block layout
//   Lane         how a lane of a SIMD group splits a block, and the
//                registers it loads the block's activations into
//   lane_of      the lane map, from the lane's index in its SIMD group
//   first_block, the blocks a lane walks: its first block and the stride
//   block_stride
//   values_per_block  the values one block holds: the width every body's
//                block-count arithmetic reads (the pointer walks, the norm's
//                count) — QK_K for Q4_K, Q5_K and Q6_K, 64 for MLX's groups,
//                32 for Q8_0
//   load<LOAD>   its input load — plain, the norm on load (the gain and the
//                sum of squares), or the route-weight form
//   partial      the dot of one block against the loaded input
//   fold_blocks, the folded down's K' in the quant's blocks, and the slot
//   fold_slot    a block of K' belongs to; `head_scales`, `sums_finite` and
//                `partial_head` serve the head-normed row loop, `value` the
//                embedding, and `dequant16` the prompt GEMMs' staging
//
// Every kernel body over quantized weights — the mat-vec's row loop with its
// modes, the expert pair's SwiGLU, the folded down, the head's argmax, the
// embedding — is a template over this; the entry points are the
// instantiations, under their host names. A new weight format adds one owner
// here and instantiates the bodies; no body is copied per quant.
//
// The lane maps (Q4_K's ix/iq/ir, Q6_K's tid/ix/ip/il, Q8_0's ix/il, Q5_K's
// tid/ix/iq/ir), the block strides, the uint16 nibble trick, the 6-bit
// unpacking, Q5_K's fifth-bit masks and the sum associations are
// llama.cpp's, from its kernel_mul_mv_q4_K_f32, kernel_mul_mv_q6_K_f32,
// kernel_mul_mv_q8_0_f32 and kernel_mul_mv_q5_K_f32 (MIT licence; see
// THIRD_PARTY_NOTICES.md):
//   ggml/src/ggml-metal/kernels/mul_mv.metal
// The same map was the dense mat-vec's, the pair's and the fold's since each
// landed; this file is those copies folded into one. A4's map is ours, on
// the same idiom (the nibble trick above), over MLX's group layout; A8's is
// A4's over its one-byte codes. A4 and A8 each have one decode rule over
// the storage of a group's scale and bias (`AffineBf16`, `AffineF16`): A4 and
// A8 are the rule over BF16 bits, A4H and A8H the same rule over F16 bits.
//
// This file comes after norm.metal and before every kernel file that reads
// quantized weights; it defines what they use of it (QK_K, FOR_UNROLL and
// the block structs).

#include <metal_stdlib>

using namespace metal;

#pragma METAL fp contract(off)

#define QK_K 256
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

struct block_q4_K {
    half    d;
    half    dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};

static_assert(sizeof(block_q4_K) == 144, "block_q4_K is 144 bytes");

struct block_q6_K {
    uint8_t ql[128];
    uint8_t qh[64];
    char    scales[16];
    half    d;
};

static_assert(sizeof(block_q6_K) == 210, "block_q6_K is 210 bytes");

// MLX's affine 4-bit group (crates/gguf's MLX_A4 and MLX_A4H): 64 values, 36
// bytes — eight u32 words of codes, then a scale and a bias, BF16 bits in
// MLX_A4 and F16 bits in MLX_A4H. Value 8w + j is the 4 bits at bit 4j of
// word w, the low nibble first.
struct block_mlx_a4 {
    uint32_t qs[8];
    uint16_t scale;  // BF16 or F16 bits, as the type stores them
    uint16_t bias;   // BF16 or F16 bits, as the type stores them
};

static_assert(sizeof(block_mlx_a4) == 36, "block_mlx_a4 is 36 bytes");

// MLX's affine 8-bit group (crates/gguf's MLX_A8 and MLX_A8H): 64 values, 68
// bytes — sixteen u32 words of one-byte codes, then a scale and a bias, BF16
// bits in MLX_A8 and F16 bits in MLX_A8H. Value 4w + j is byte j of word w.
struct block_mlx_a8 {
    uint32_t qs[16];
    uint16_t scale;  // BF16 or F16 bits, as the type stores them
    uint16_t bias;   // BF16 or F16 bits, as the type stores them
};

static_assert(sizeof(block_mlx_a8) == 68, "block_mlx_a8 is 68 bytes");

struct block_q8_0 {
    half   d;
    int8_t qs[32];
};

static_assert(sizeof(block_q8_0) == 34, "block_q8_0 is 34 bytes");

// Q4_K's d, dmin and 6-bit scale/min pairs, then the fifth bit of every
// value (value 64j + l takes bit 2j of qh[l], value 64j + 32 + l bit
// 2j + 1) and Q4_K's nibbles.
struct block_q5_K {
    half    d;
    half    dmin;
    uint8_t scales[12];
    uint8_t qh[32];
    uint8_t qs[128];
};

static_assert(sizeof(block_q5_K) == 176, "block_q5_K is 176 bytes");

// What a load does to each value it loads.
enum LoadMode {
    // The store, residual and head-norm mat-vecs: the value as it is.
    LOAD_PLAIN,
    // The norm on load (norm.metal): its square joins the lane's sum of
    // squares, then the gain multiplies it.
    LOAD_NORM,
    // The fold: the route weight of the block's slot multiplies it.
    LOAD_ROUTE,
};

// Four consecutive activations (or gains) in one load. The address is a
// multiple of 16 bytes: every lane offset below is a multiple of four values,
// and each launch refuses an input or gain window that does not start on 16
// bytes.
inline float4 act4(device const float * p) {
    return *(device const float4 *)p;
}

// The four 16-bit words of a Q4_K-packed scale table's half `iq`: the eight
// 6-bit scales and mins of that half's four 32-value groups, as bytes
// [scale 2iq, scale 2iq+1, min 2iq, min 2iq+1, scale 2iq+4, scale 2iq+5,
// min 2iq+4, min 2iq+5]. `sc` is the table read as words, offset by `iq`.
inline void k4_scale_words(device const uint16_t * sc, thread uint16_t * sc16) {
    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;
    sc16[0] = sc[0] & kmask1;
    sc16[1] = sc[2] & kmask1;
    sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
    sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);
}

struct Q4K {
    typedef block_q4_K Block;

    // The lane's map and registers: ix picks the lane's blocks (four apart),
    // iq a sub-block pair of the block, ir a quarter of the sub-block. The
    // registers are the loaded activations: yl and yh the two 64-value
    // groups the lane's quarter of the sub-block pair reads, sumy their
    // sums (the dmin term's).
    struct Lane {
        short   ix;
        short   iq;
        short   ir;
        float   yl[16];
        float   yh[16];
        float4  sumy;
    };

    static inline Lane lane_of(ushort tiisg) {
        Lane lane;
        lane.ix = (short)(tiisg / 8);  // 0...3
        const short it = (short)(tiisg % 8);  // 0...7
        lane.iq = (short)(it / 4);      // 0 or 1
        lane.ir = (short)(it % 4);      // 0...3
        return lane;
    }

    static inline int first_block(const thread Lane &lane) {
        return lane.ix;
    }

    static inline int block_stride() {
        return 4;
    }

    // The values one block holds: the width every body's block-count
    // arithmetic reads (the pointer walks, the norm's count).
    static inline int values_per_block() {
        return QK_K;
    }

    // The lane's values' offset in a block: its quarter of the sub-block
    // pair its iq selects.
    static inline int input_offset(const thread Lane &lane) {
        return 64 * lane.iq + 8 * lane.ir;
    }

    // The activations of one block, in the lane's registers. `gain` (the
    // block's gain) and `sum_sq` (the lane's sum of squares) are read by the
    // norm load only, `scale` by the route load; the others pass null for
    // both.
    template <LoadMode LOAD>
    static inline void load(thread Lane &lane,
                            device const float * block,
                            device const float * gain,
                            float scale,
                            thread float * sum_sq) {
        const int base = input_offset(lane);
        lane.sumy = float4(0.f, 0.f, 0.f, 0.f);
        FOR_UNROLL (short j = 0; j < 8; j += 4) {
            const float4 x0 = act4(block + base + j +   0);
            const float4 x1 = act4(block + base + j +  32);
            const float4 x2 = act4(block + base + j + 128);
            const float4 x3 = act4(block + base + j + 160);
            float4 g0 = float4(1.f);
            float4 g1 = float4(1.f);
            float4 g2 = float4(1.f);
            float4 g3 = float4(1.f);
            if (LOAD == LOAD_NORM) {
                g0 = act4(gain + base + j +   0);
                g1 = act4(gain + base + j +  32);
                g2 = act4(gain + base + j + 128);
                g3 = act4(gain + base + j + 160);
            }
            FOR_UNROLL (short k = 0; k < 4; ++k) {
                const short i = j + k;
                float a0 = x0[k];
                float a1 = x1[k];
                float a2 = x2[k];
                float a3 = x3[k];
                if (LOAD == LOAD_ROUTE) {
                    a0 *= scale;
                    a1 *= scale;
                    a2 *= scale;
                    a3 *= scale;
                } else if (LOAD == LOAD_NORM) {
                    *sum_sq += (a0 * a0 + a1 * a1) + (a2 * a2 + a3 * a3);
                    a0 *= g0[k];
                    a1 *= g1[k];
                    a2 *= g2[k];
                    a3 *= g3[k];
                }
                lane.yl[i + 0] = a0;
                lane.sumy[0] += a0;
                lane.yl[i + 8] = a1;
                lane.sumy[1] += a1;
                lane.yh[i + 0] = a2;
                lane.sumy[2] += a2;
                lane.yh[i + 8] = a3;
                lane.sumy[3] += a3;
            }
        }
    }

    // One block's contribution. The subtraction keeps llama.cpp's
    // association: d times the nibble sum, minus dmin times the activation
    // sum.
    static inline float partial(device const Block & blk,
                                const thread Lane & lane) {
        uint16_t sc16[4];
        thread const uint8_t * sc8 = (thread const uint8_t *)sc16;
        device const uint16_t * sc = (device const uint16_t *)blk.scales + lane.iq;
        device const uint16_t * q1 = (device const uint16_t *)blk.qs + 16 * lane.iq + 4 * lane.ir;
        device const half * dh = &blk.d;

        k4_scale_words(sc, sc16);

        device const uint16_t * q2 = q1 + 32;
        float4 acc1 = {0.f, 0.f, 0.f, 0.f};
        float4 acc2 = {0.f, 0.f, 0.f, 0.f};
        FOR_UNROLL (short i = 0; i < 4; ++i) {
            acc1[0] += lane.yl[2 * i + 0] * (q1[i] & 0x000F);
            acc1[1] += lane.yl[2 * i + 1] * (q1[i] & 0x0F00);
            acc1[2] += lane.yl[2 * i + 8] * (q1[i] & 0x00F0);
            acc1[3] += lane.yl[2 * i + 9] * (q1[i] & 0xF000);
            acc2[0] += lane.yh[2 * i + 0] * (q2[i] & 0x000F);
            acc2[1] += lane.yh[2 * i + 1] * (q2[i] & 0x0F00);
            acc2[2] += lane.yh[2 * i + 8] * (q2[i] & 0x00F0);
            acc2[3] += lane.yh[2 * i + 9] * (q2[i] & 0xF000);
        }

        return dh[0] * ((acc1[0] + 1.f/256.f * acc1[1]) * sc8[0] +
                        (acc1[2] + 1.f/256.f * acc1[3]) * sc8[1] * 1.f/16.f +
                        (acc2[0] + 1.f/256.f * acc2[1]) * sc8[4] +
                        (acc2[2] + 1.f/256.f * acc2[3]) * sc8[5] * 1.f/16.f)
             - dh[1] * (lane.sumy[0] * sc8[2] + lane.sumy[1] * sc8[3] +
                        lane.sumy[2] * sc8[6] + lane.sumy[3] * sc8[7]);
    }

    // The same block dotted against a head-scaled input, in the gated delta
    // net's two-halves form: the block splits into two 128-value head
    // groups, the scale of each multiplies its whole term once, not every
    // value in it (the dot is linear in the input), and the dmin subtraction
    // is per half. Q6_K has no entry point in this mode.
    static inline float partial_head(device const Block & blk,
                                     const thread Lane & lane,
                                     float lo,
                                     float hi) {
        uint16_t sc16[4];
        thread const uint8_t * sc8 = (thread const uint8_t *)sc16;
        device const uint16_t * sc = (device const uint16_t *)blk.scales + lane.iq;
        device const uint16_t * q1 = (device const uint16_t *)blk.qs + 16 * lane.iq + 4 * lane.ir;
        device const half * dh = &blk.d;

        k4_scale_words(sc, sc16);

        device const uint16_t * q2 = q1 + 32;
        float4 acc1 = {0.f, 0.f, 0.f, 0.f};
        float4 acc2 = {0.f, 0.f, 0.f, 0.f};
        FOR_UNROLL (short i = 0; i < 4; ++i) {
            acc1[0] += lane.yl[2 * i + 0] * (q1[i] & 0x000F);
            acc1[1] += lane.yl[2 * i + 1] * (q1[i] & 0x0F00);
            acc1[2] += lane.yl[2 * i + 8] * (q1[i] & 0x00F0);
            acc1[3] += lane.yl[2 * i + 9] * (q1[i] & 0xF000);
            acc2[0] += lane.yh[2 * i + 0] * (q2[i] & 0x000F);
            acc2[1] += lane.yh[2 * i + 1] * (q2[i] & 0x0F00);
            acc2[2] += lane.yh[2 * i + 8] * (q2[i] & 0x00F0);
            acc2[3] += lane.yh[2 * i + 9] * (q2[i] & 0xF000);
        }

        return lo * (dh[0] * ((acc1[0] + 1.f/256.f * acc1[1]) * sc8[0] +
                              (acc1[2] + 1.f/256.f * acc1[3]) * sc8[1] * 1.f/16.f) -
                     dh[1] * (lane.sumy[0] * sc8[2] + lane.sumy[1] * sc8[3])) +
               hi * (dh[0] * ((acc2[0] + 1.f/256.f * acc2[1]) * sc8[4] +
                              (acc2[2] + 1.f/256.f * acc2[3]) * sc8[5] * 1.f/16.f) -
                     dh[1] * (lane.sumy[2] * sc8[6] + lane.sumy[3] * sc8[7]));
    }

    // The head-normed row loop's asks, per quant. `head_scales`: the block's
    // two 128-value head groups' scales. `sums_finite`: the lane's group
    // sums all finite. The folded down's: the blocks of one slot of the
    // fold's K', and the slot a block of K' belongs to.
    static inline float2 head_scales(threadgroup const float * scales, int ib) {
        return float2(scales[2 * ib], scales[2 * ib + 1]);
    }

    static inline bool sums_finite(const thread Lane &lane) {
        return isfinite(lane.sumy[0]) && isfinite(lane.sumy[1]) &&
               isfinite(lane.sumy[2]) && isfinite(lane.sumy[3]);
    }

    static inline int fold_blocks() {
        return BLOOMERY_EXPERT_SLOTS * (BLOOMERY_EXPERT_K / values_per_block());
    }

    static inline int fold_slot(int ib) {
        return ib / (BLOOMERY_EXPERT_K / values_per_block());
    }

    // Sixteen consecutive values of a block, [16*il, 16*il + 16) in natural
    // order (value 16*il + j at reg[j/4][j%4]), the GEMM's A staging: the
    // 4x4 of the operand type one thread holds. The scale and the min are
    // the owner's unpack (the same sc16 words `partial` reads, over the u16
    // window the chunk's 32-value group lives in), and the value is the
    // owner's too: the group's scale times the nibble, less the dmin term.
    // The group is `il >> 1`, its low nibbles under d and its high under
    // d/16 — the layout the lane map above walks, which is llama.cpp's
    // dequantize_q4_K's shape (dequantize.h); no second decode lives here.
    template <typename T>
    static inline void dequant16(device const Block & blk,
                                 short il,
                                 thread matrix<T, 4, 4> & reg) {
        const short iq = (il >> 2) & 1;
        uint16_t sc16[4];
        thread const uint8_t * sc8 = (thread const uint8_t *)sc16;
        device const uint16_t * sc = (device const uint16_t *)blk.scales + iq;
        k4_scale_words(sc, sc16);
        // sc8 holds the half's four scale/min pairs: [scale 2iq, scale 2iq+1,
        // min 2iq, min 2iq+1, scale 2iq+4, scale 2iq+5, min 2iq+4, min 2iq+5].
        const short at = (il >> 1 & 1) + 4 * (il >> 3);
        const float d = (il & 2) == 0 ? (float)blk.d : (float)blk.d / 16.f;
        const float dl = d * (float)sc8[at];
        const float ml = (float)blk.dmin * (float)sc8[2 + at];

        device const uint16_t * q = (device const uint16_t *)(blk.qs + 32 * (il >> 2) + 16 * (il & 1));
        const ushort mask = (il & 2) == 0 ? 0x0F0F : 0xF0F0;
        FOR_UNROLL(short i = 0; i < 8; ++i) {
            const uchar2 pair = as_type<uchar2>((ushort)(q[i] & mask));
            reg[(2 * i) / 4][(2 * i) % 4] = dl * (float)pair.x - ml;
            reg[(2 * i + 1) / 4][(2 * i + 1) % 4] = dl * (float)pair.y - ml;
        }
    }
};

struct Q6K {
    typedef block_q6_K Block;

    // The lane's map and registers: ix picks the lane's blocks (two apart),
    // ip the half of the block, il the eighth of the half; l0 is that
    // eighth's first value. yl holds the loaded activations; the scales of
    // the 6-bit values cover the whole block, so no group sums are needed.
    struct Lane {
        short   ix;
        short   ip;
        short   l0;
        float   yl[16];
    };

    static inline Lane lane_of(ushort tiisg) {
        Lane lane;
        const short tid = (short)(tiisg / 2);
        lane.ix = (short)(tiisg % 2);
        lane.ip = (short)(tid / 8);   // 0 or 1
        const short il = (short)(tid % 8);
        lane.l0 = (short)(4 * il);
        return lane;
    }

    static inline int first_block(const thread Lane &lane) {
        return lane.ix;
    }

    static inline int block_stride() {
        return 2;
    }

    static inline int values_per_block() {
        return QK_K;
    }

    // The lane's values' offset in a block: its eighth of the half.
    static inline int input_offset(const thread Lane &lane) {
        return 128 * lane.ip + lane.l0;
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
        const float4 x0 = act4(block + base +  0);
        const float4 x1 = act4(block + base + 32);
        const float4 x2 = act4(block + base + 64);
        const float4 x3 = act4(block + base + 96);
        float4 g0 = float4(1.f);
        float4 g1 = float4(1.f);
        float4 g2 = float4(1.f);
        float4 g3 = float4(1.f);
        if (LOAD == LOAD_NORM) {
            g0 = act4(gain + base +  0);
            g1 = act4(gain + base + 32);
            g2 = act4(gain + base + 64);
            g3 = act4(gain + base + 96);
        }
        FOR_UNROLL (short l = 0; l < 4; ++l) {
            float a0 = x0[l];
            float a1 = x1[l];
            float a2 = x2[l];
            float a3 = x3[l];
            if (LOAD == LOAD_ROUTE) {
                a0 *= scale;
                a1 *= scale;
                a2 *= scale;
                a3 *= scale;
            } else if (LOAD == LOAD_NORM) {
                *sum_sq += (a0 * a0 + a1 * a1) + (a2 * a2 + a3 * a3);
                a0 *= g0[l];
                a1 *= g1[l];
                a2 *= g2[l];
                a3 *= g3[l];
            }
            lane.yl[4 * l + 0] = a0;
            lane.yl[4 * l + 1] = a1;
            lane.yl[4 * l + 2] = a2;
            lane.yl[4 * l + 3] = a3;
        }
    }

    // One block's contribution. The high bits of each 6-bit value are split
    // out before the offset, so the offset stays on the assembled value.
    static inline float partial(device const Block & blk,
                                const thread Lane & lane) {
        constexpr uint8_t kmask1 = 0x03;
        constexpr uint8_t kmask2 = 0x0C;
        constexpr uint8_t kmask3 = 0x30;
        constexpr uint8_t kmask4 = 0xC0;
        const short q_offset_l = 64 * lane.ip + lane.l0;
        const short q_offset_h = 32 * lane.ip + lane.l0;
        const short is = 8 * lane.ip + lane.l0 / 16;

        device const uint8_t * q1 = blk.ql + q_offset_l;
        device const uint8_t * q2 = q1 + 32;
        device const uint8_t * qh = blk.qh + q_offset_h;
        device const char    * sc = blk.scales + is;
        device const half    * dh = &blk.d;
        float4 sums = {0.f, 0.f, 0.f, 0.f};
        FOR_UNROLL (short l = 0; l < 4; ++l) {
            uint8_t h0 = (uint8_t)((qh[l] & kmask1) << 4);
            uint8_t h1 = (uint8_t)((qh[l] & kmask2) << 2);
            uint8_t h2 = (uint8_t)((qh[l] & kmask3) << 0);
            uint8_t h3 = (uint8_t)((qh[l] & kmask4) >> 2);
            sums[0] += lane.yl[4 * l + 0] * ((char)((q1[l] & 0xF) | h0) - 32);
            sums[1] += lane.yl[4 * l + 1] * ((char)((q2[l] & 0xF) | h1) - 32);
            sums[2] += lane.yl[4 * l + 2] * ((char)((q1[l]  >> 4) | h2) - 32);
            sums[3] += lane.yl[4 * l + 3] * ((char)((q2[l]  >> 4) | h3) - 32);
        }
        return dh[0] * (sums[0] * sc[0] + sums[1] * sc[2] + sums[2] * sc[4] + sums[3] * sc[6]);
    }

    static inline int fold_blocks() {
        return BLOOMERY_EXPERT_SLOTS * (BLOOMERY_EXPERT_K / values_per_block());
    }

    static inline int fold_slot(int ib) {
        return ib / (BLOOMERY_EXPERT_K / values_per_block());
    }

    // Sixteen consecutive values of a block, [16*il, 16*il + 16) in natural
    // order (value 16*il + j at reg[j/4][j%4]), the GEMM's A staging: the
    // 4x4 of the operand type one thread holds. A chunk sits whole inside
    // one nibble side of its 128-value half: its ql bytes are the half's
    // (h0 & 63) run, low nibbles before value 64 of the half and high
    // after; its high bits are the (h0 & 31) run of qh at twice the
    // 32-value block's place; its scale is `scales[il]`, the same one word
    // `partial` reads. The value is the owner's: d * sc * (q - 32), the
    // product d * sc rounded once and the (q - 32) multiple of it rounded
    // once, which `fma(q, d * sc, -32 * d * sc)` is exactly (the product of
    // the integer and d * sc is the one rounding, the -32 term exact). The
    // runs are read as 16-bit words (a block is 210 bytes, so two bytes is
    // all the alignment a run has): two codes of six bits sit in the two
    // bytes of the word the masks and the shift assemble.
    template <typename T>
    static inline void dequant16(device const Block & blk,
                                 short il,
                                 thread matrix<T, 4, 4> & reg) {
        const short ip = il >> 3;
        const short h0 = (il & 7) * 16;
        device const uint16_t * ql = (device const uint16_t *)(blk.ql + 64 * ip + (h0 & 63));
        device const uint16_t * qh = (device const uint16_t *)(blk.qh + 32 * ip + (h0 & 31));
        const ushort shift = (ushort)(2 * (h0 >> 5));
        const ushort low_shift = (h0 & 64) == 0 ? 0 : 4;
        const float dsc = (float)blk.d * (float)blk.scales[il];
        const float offset = -32.f * dsc;
        FOR_UNROLL(short i = 0; i < 8; ++i) {
            const ushort lo = (ushort)((ql[i] >> low_shift) & 0x0F0F);
            const ushort hi = (ushort)((qh[i] >> shift) & 0x0303);
            const uchar2 codes = as_type<uchar2>((ushort)(lo | (hi << 4)));
            reg[(2 * i) / 4][(2 * i) % 4] = fma((float)codes.x, dsc, offset);
            reg[(2 * i + 1) / 4][(2 * i + 1) % 4] = fma((float)codes.y, dsc, offset);
        }
    }
};

// How an affine group stores its scale and bias, the one place the pair is
// widened to f32: both BF16 bits (MLX_A4, MLX_A8) or both F16 bits (MLX_A4H,
// MLX_A8H). Either widens exactly. `widen` takes one 16-bit field; the
// `_of_word` forms take the 32-bit word the pair is read as (the scale in the
// low half, the bias in the high half, little-endian).
struct AffineBf16 {
    static inline float widen(uint bits) {
        return as_type<float>(bits << 16);
    }

    static inline float scale_of_word(uint32_t sb) {
        return as_type<float>(sb << 16);
    }

    static inline float bias_of_word(uint32_t sb) {
        return as_type<float>(sb & 0xFFFF0000u);
    }
};

struct AffineF16 {
    static inline float widen(uint bits) {
        return float(as_type<half>((ushort)bits));
    }

    static inline float scale_of_word(uint32_t sb) {
        return float(as_type<half2>(sb).x);
    }

    static inline float bias_of_word(uint32_t sb) {
        return float(as_type<half2>(sb).y);
    }
};

// MLX's affine 4-bit decode owner (block_mlx_a4, above): one 64-value group
// is one block, so a block count a body reads is a group count and the lane
// map splits one group across four lanes — a SIMD group's 32 lanes cover
// eight groups, 512 values, a pass (the stride). The lane's two consecutive
// code words hold its sixteen values in address order (nibble j of a word is
// value j of the word), so the uint16 mask trick, over the four halves of the
// word pair, reads each nibble into its own accumulator and the shift of a
// whole accumulator folds once into an exact power of two. MLX's own
// `affine_qmv_fast` (quantized.h) is the design reference for the bias fold;
// its four-rows-a-thread x reuse is this engine's SIMD-group row loop instead.
// The rule is a template over how the group's pair is stored (`Pair`): A4 is
// it over BF16 bits and A4H over F16 bits.
template <typename Pair>
struct A4T {
    typedef block_mlx_a4 Block;

    // The lane's map and registers: ix picks the lane's groups (eight apart),
    // iq the quarter of one. yl holds the loaded activations — sixteen
    // consecutive values of the group, the two code words they sit in — and
    // sumy their sum, the bias fold's.
    struct Lane {
        short   ix;
        short   iq;
        float   yl[16];
        float   sumy;
    };

    static inline Lane lane_of(ushort tiisg) {
        Lane lane;
        lane.ix = (short)(tiisg / 4);  // 0...7
        lane.iq = (short)(tiisg % 4);  // 0...3
        return lane;
    }

    static inline int first_block(const thread Lane &lane) {
        return lane.ix;
    }

    static inline int block_stride() {
        return 8;
    }

    static inline int values_per_block() {
        return 64;
    }

    // The lane's values' offset in a group: its quarter.
    static inline int input_offset(const thread Lane &lane) {
        return 16 * lane.iq;
    }

    // The activations of one group, in the lane's registers. `gain` (the
    // group's gain) and `sum_sq` are read by the norm load only, `scale` by
    // the route load; the others pass null for both.
    template <LoadMode LOAD>
    static inline void load(thread Lane &lane,
                            device const float * block,
                            device const float * gain,
                            float scale,
                            thread float * sum_sq) {
        const int base = input_offset(lane);
        lane.sumy = 0.f;
        FOR_UNROLL (short l = 0; l < 4; ++l) {
            const float4 xq = act4(block + base + 4 * l);
            float4 gq = float4(1.f);
            if (LOAD == LOAD_NORM) {
                gq = act4(gain + base + 4 * l);
            }
            float a0 = xq.x;
            float a1 = xq.y;
            float a2 = xq.z;
            float a3 = xq.w;
            if (LOAD == LOAD_ROUTE) {
                a0 *= scale;
                a1 *= scale;
                a2 *= scale;
                a3 *= scale;
            } else if (LOAD == LOAD_NORM) {
                *sum_sq += (a0 * a0 + a1 * a1) + (a2 * a2 + a3 * a3);
                a0 *= gq.x;
                a1 *= gq.y;
                a2 *= gq.z;
                a3 *= gq.w;
            }
            lane.yl[4 * l + 0] = a0;
            lane.yl[4 * l + 1] = a1;
            lane.yl[4 * l + 2] = a2;
            lane.yl[4 * l + 3] = a3;
            lane.sumy += (a0 + a1) + (a2 + a3);
        }
    }

    static inline bool sums_finite(const thread Lane &lane) {
        return isfinite(lane.sumy);
    }

    // The group's pair, widened as `Pair` stores it: exact.
    static inline float scale_of(device const Block & blk) {
        return Pair::widen(blk.scale);
    }

    static inline float bias_of(device const Block & blk) {
        return Pair::widen(blk.bias);
    }

    // One group's contribution. The scale multiplies the code dot and the
    // bias the lane's input sum, as MLX's `qdot` folds its bias and Q4_K its
    // dmin term. The lane's two code words and the scale-and-bias pair (the
    // two 16-bit fields sit at byte 32, scale in the low half) are read as
    // 32-bit words, the same bits as 16-bit reads in fewer, wider loads; a
    // launch checks that the weights start on a 4-byte boundary.
    static inline float partial(device const Block & blk,
                                const thread Lane & lane) {
        device const uint32_t * q = blk.qs + 2 * lane.iq;
        const uint32_t sb = *(device const uint32_t *)&blk.scale;
        const uint32_t w0 = q[0];
        const uint32_t w1 = q[1];
        const uint32_t words[4] = {w0 & 0xFFFFu, w0 >> 16, w1 & 0xFFFFu, w1 >> 16};
        float4 acc = {0.f, 0.f, 0.f, 0.f};
        FOR_UNROLL (short m = 0; m < 4; ++m) {
            acc[0] += lane.yl[4 * m + 0] * (words[m] & 0x000F);
            acc[1] += lane.yl[4 * m + 1] * (words[m] & 0x00F0);
            acc[2] += lane.yl[4 * m + 2] * (words[m] & 0x0F00);
            acc[3] += lane.yl[4 * m + 3] * (words[m] & 0xF000);
        }
        return Pair::scale_of_word(sb) * ((acc[0] + 1.f/16.f * acc[1]) +
                                          (1.f/256.f * acc[2] + 1.f/4096.f * acc[3]))
             + Pair::bias_of_word(sb) * lane.sumy;
    }

    // A 64-value group sits whole inside one 128-column value head, so the
    // head's one scale multiplies the whole contribution; `hi` is that same
    // head's and goes unused.
    static inline float partial_head(device const Block & blk,
                                     const thread Lane & lane,
                                     float lo,
                                     float hi) {
        return lo * partial(blk, lane);
    }

    static inline float2 head_scales(threadgroup const float * scales, int ib) {
        const int head = ib >> 1;
        return float2(scales[head], scales[head]);
    }

    static inline int fold_blocks() {
        return BLOOMERY_EXPERT_SLOTS * (BLOOMERY_EXPERT_K / values_per_block());
    }

    static inline int fold_slot(int ib) {
        return ib / (BLOOMERY_EXPERT_K / values_per_block());
    }

    // Value `j` of a group, the host reader's decode: `s·q + b`, the product
    // and the sum each rounded once (the fp-contract pragma holds the order),
    // the bits the embed's clause holds it to. The code word is read through
    // 16-bit words: the embed's and the GEMMs' launches check a 2-byte weights
    // alignment, and a wider read would ask for 4.
    static inline float value(device const Block & blk, uint j) {
        device const uint16_t * w = (device const uint16_t *)blk.qs;
        const uint word = (uint)w[2 * (j / 8)] | ((uint)w[2 * (j / 8) + 1] << 16);
        const float q = float((word >> (4 * (j % 8))) & 0xF);
        return scale_of(blk) * q + bias_of(blk);
    }

    // Sixteen consecutive values of a group, [16*il, 16*il + 16) in natural
    // order (value 16*il + j at reg[j/4][j%4]), the GEMM's A staging: the
    // 4x4 of the operand type one thread holds, `value`'s bits in fewer
    // instructions. A 16-bit code word holds four values; value t of it is
    // its nibble t masked in place, q·16^t (exact as a float), times the
    // scale over 16^t (exact: a power of two), plus the bias in one fma.
    // The product q·s is exact (a 4-bit code by a BF16's 8-bit or an F16's
    // 11-bit significand), so the fma's one rounding is `value`'s add and the
    // f32 sum is `value`'s;
    // it is rounded once into the operand type.
    template <typename T>
    static inline void dequant16(device const Block & blk,
                                 short il,
                                 thread matrix<T, 4, 4> & reg) {
        device const uint16_t * w = (device const uint16_t *)blk.qs + 4 * il;
        const float s0 = scale_of(blk);
        const float b = bias_of(blk);
        const float s1 = s0 * (1.f / 16.f);
        const float s2 = s0 * (1.f / 256.f);
        const float s3 = s0 * (1.f / 4096.f);
        FOR_UNROLL(short h = 0; h < 4; ++h) {
            const uint word = (uint)w[h];
            reg[h][0] = fma((float)(word & 0x000Fu), s0, b);
            reg[h][1] = fma((float)(word & 0x00F0u), s1, b);
            reg[h][2] = fma((float)(word & 0x0F00u), s2, b);
            reg[h][3] = fma((float)(word & 0xF000u), s3, b);
        }
    }
};

typedef A4T<AffineBf16> A4;
typedef A4T<AffineF16> A4H;

// MLX's affine 8-bit decode owner (block_mlx_a8, above): A4's lane map and
// registers over the one-byte codes — a block is one 64-value group, the
// lane's quarter of it four code words, its sixteen values the words' bytes
// in address order, so a word unpacks as one uchar4 vector cast (the code is
// the byte; no nibble step). The dot keeps A4's shape — four accumulators
// combined pairwise, the scale on the code dot and the bias on the lane's
// input sum — which is MLX's own qdot at 8 bits (quantized.h: the codes read
// as bytes, `scale * accum + sum * bias`). The registers, the load, the group
// sums, the head scales and the fold counts are A4's own: the affine
// layout's, not the code width's.
template <typename Pair>
struct A8T {
    typedef block_mlx_a8 Block;
    typedef A4T<Pair> Base;
    typedef typename Base::Lane Lane;

    static inline Lane lane_of(ushort tiisg) {
        return Base::lane_of(tiisg);
    }

    static inline int first_block(const thread Lane &lane) {
        return Base::first_block(lane);
    }

    static inline int block_stride() {
        return Base::block_stride();
    }

    static inline int values_per_block() {
        return 64;
    }

    // The lane's values' offset in a group: its quarter.
    static inline int input_offset(const thread Lane &lane) {
        return Base::input_offset(lane);
    }

    // The activations of one group, in the lane's registers: A4's load, the
    // lane's sixteen values and their sum the bias fold reads. `gain` (the
    // group's gain) and `sum_sq` are read by the norm load only, `scale` by
    // the route load; the others pass null for both.
    template <LoadMode LOAD>
    static inline void load(thread Lane &lane,
                            device const float * block,
                            device const float * gain,
                            float scale,
                            thread float * sum_sq) {
        Base::template load<LOAD>(lane, block, gain, scale, sum_sq);
    }

    static inline bool sums_finite(const thread Lane &lane) {
        return Base::sums_finite(lane);
    }

    // The group's pair, widened as `Pair` stores it: exact.
    static inline float scale_of(device const Block & blk) {
        return Pair::widen(blk.scale);
    }

    static inline float bias_of(device const Block & blk) {
        return Pair::widen(blk.bias);
    }

    // One group's contribution. The scale multiplies the code dot and the
    // bias the lane's input sum, as A4's and MLX's `qdot` fold theirs. The
    // lane's four code words and the scale-and-bias pair (the two 16-bit
    // fields sit at byte 64, scale in the low half) are read as 32-bit words,
    // the codes unpacked as uchar4 vector casts into float4; a launch checks
    // that the weights start on a 4-byte boundary.
    static inline float partial(device const Block & blk,
                                const thread Lane & lane) {
        device const uint32_t * q = blk.qs + 4 * lane.iq;
        const uint32_t sb = *(device const uint32_t *)&blk.scale;
        float4 acc = {0.f, 0.f, 0.f, 0.f};
        FOR_UNROLL (short m = 0; m < 4; ++m) {
            const float4 codes = float4(as_type<uchar4>(q[m]));
            acc[0] += lane.yl[4 * m + 0] * codes[0];
            acc[1] += lane.yl[4 * m + 1] * codes[1];
            acc[2] += lane.yl[4 * m + 2] * codes[2];
            acc[3] += lane.yl[4 * m + 3] * codes[3];
        }
        const float code_dot = (acc[0] + acc[1]) + (acc[2] + acc[3]);
        const float s = Pair::scale_of_word(sb);
        const float b = Pair::bias_of_word(sb);
        return s * code_dot + b * lane.sumy;
    }

    // A 64-value group sits whole inside one 128-column value head, so the
    // head's one scale multiplies the whole contribution; `hi` is that same
    // head's and goes unused.
    static inline float partial_head(device const Block & blk,
                                     const thread Lane & lane,
                                     float lo,
                                     float hi) {
        return lo * partial(blk, lane);
    }

    static inline float2 head_scales(threadgroup const float * scales, int ib) {
        return Base::head_scales(scales, ib);
    }

    static inline int fold_blocks() {
        return Base::fold_blocks();
    }

    static inline int fold_slot(int ib) {
        return Base::fold_slot(ib);
    }

    // Value `j` of a group, the host reader's decode: `s·q + b`, the product
    // exact (an 8-bit code by a BF16's 8-bit or an F16's 11-bit significand)
    // and the sum rounded once, the bits the embed's clause holds it to. The
    // code is the group's byte `j`.
    static inline float value(device const Block & blk, uint j) {
        device const uint8_t * q = (device const uint8_t *)blk.qs;
        return scale_of(blk) * float(q[j]) + bias_of(blk);
    }

    // Sixteen consecutive values of a group, [16*il, 16*il + 16) in natural
    // order (value 16*il + j at reg[j/4][j%4]), the GEMM's A staging: the
    // owner's own `value` in fewer instructions, one 16-bit word holding two
    // codes as its bytes and each fma's product exact, so the fma's one
    // rounding is `value`'s add and the f32 sum is `value`'s; it is rounded
    // once into the operand type.
    template <typename T>
    static inline void dequant16(device const Block & blk,
                                 short il,
                                 thread matrix<T, 4, 4> & reg) {
        device const uint16_t * w = (device const uint16_t *)blk.qs + 8 * il;
        const float s = scale_of(blk);
        const float bias = bias_of(blk);
        FOR_UNROLL(short h = 0; h < 8; ++h) {
            const uchar2 codes = as_type<uchar2>(w[h]);
            reg[(2 * h) / 4][(2 * h) % 4] = fma((float)codes.x, s, bias);
            reg[(2 * h + 1) / 4][(2 * h + 1) % 4] = fma((float)codes.y, s, bias);
        }
    }
};

typedef A8T<AffineBf16> A8;
typedef A8T<AffineF16> A8H;

// GGUF's Q8_0 decode owner (block_q8_0, above): a 32-value block of signed
// bytes under one f16 scale, `y = q·d`. The lane map is llama.cpp's
// kernel_mul_mv_q8_0_f32's inside one SIMD group: four lanes split a block
// into quarters of eight values, so the 32 lanes cover eight blocks, 256
// values, a pass (the stride). llama.cpp splits the blocks across the
// threadgroup's SIMD groups too and reduces through threadgroup memory;
// here every SIMD group owns its rows over the whole row, the row loop's
// rule. A 32-value block sits inside one 128-column value head, so the head
// form is one scale a block, as A4's.
struct Q8_0 {
    typedef block_q8_0 Block;

    // The lane's map and registers: ix picks the lane's blocks (eight
    // apart), il the quarter of one. yl holds the loaded activations, eight
    // consecutive values.
    struct Lane {
        short   ix;
        short   il;
        float   yl[8];
    };

    static inline Lane lane_of(ushort tiisg) {
        Lane lane;
        lane.ix = (short)(tiisg / 4);  // 0...7
        lane.il = (short)(tiisg % 4);  // 0...3
        return lane;
    }

    static inline int first_block(const thread Lane &lane) {
        return lane.ix;
    }

    static inline int block_stride() {
        return 8;
    }

    static inline int values_per_block() {
        return 32;
    }

    // The lane's values' offset in a block: its quarter.
    static inline int input_offset(const thread Lane &lane) {
        return 8 * lane.il;
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
        FOR_UNROLL (short l = 0; l < 8; l += 4) {
            const float4 x8 = act4(block + base + l);
            float4 g8 = float4(1.f);
            if (LOAD == LOAD_NORM) {
                g8 = act4(gain + base + l);
            }
            float a0 = x8.x;
            float a1 = x8.y;
            float a2 = x8.z;
            float a3 = x8.w;
            if (LOAD == LOAD_ROUTE) {
                a0 *= scale;
                a1 *= scale;
                a2 *= scale;
                a3 *= scale;
            } else if (LOAD == LOAD_NORM) {
                *sum_sq += (a0 * a0 + a1 * a1) + (a2 * a2 + a3 * a3);
                a0 *= g8.x;
                a1 *= g8.y;
                a2 *= g8.z;
                a3 *= g8.w;
            }
            lane.yl[l + 0] = a0;
            lane.yl[l + 1] = a1;
            lane.yl[l + 2] = a2;
            lane.yl[l + 3] = a3;
        }
    }

    // One block's contribution: the code dot, then the scale once, in
    // llama.cpp's association.
    static inline float partial(device const Block & blk,
                                const thread Lane & lane) {
        device const int8_t * qs = blk.qs + 8 * lane.il;
        float sumq = 0.f;
        FOR_UNROLL (short i = 0; i < 8; ++i) {
            sumq += qs[i] * lane.yl[i];
        }
        return sumq * (float)blk.d;
    }

    // A block sits whole inside one value head, so the head's one scale
    // multiplies the whole contribution; `hi` is that same head's.
    static inline float partial_head(device const Block & blk,
                                     const thread Lane & lane,
                                     float lo,
                                     float hi) {
        return lo * partial(blk, lane);
    }

    static_assert(BLOOMERY_DELTA_HEAD_COLS % 32 == 0,
                  "a Q8_0 block sits inside one value head");

    static inline float2 head_scales(threadgroup const float * scales, int ib) {
        const int head = ib / (BLOOMERY_DELTA_HEAD_COLS / 32);
        return float2(scales[head], scales[head]);
    }

    // The lane's loaded values sum to a finite number: no sum is kept for
    // the dot, so the head-normed row loop's ask adds them here.
    static inline bool sums_finite(const thread Lane &lane) {
        float sum = 0.f;
        for (short i = 0; i < 8; ++i) {
            sum += lane.yl[i];
        }
        return isfinite(sum);
    }

    static inline int fold_blocks() {
        return BLOOMERY_EXPERT_SLOTS * (BLOOMERY_EXPERT_K / values_per_block());
    }

    static inline int fold_slot(int ib) {
        return ib / (BLOOMERY_EXPERT_K / values_per_block());
    }

    // Value `j` of a block, the host reader's decode: the code times the
    // scale, exact in f32 (eight bits times eleven), so the embed's bits are
    // the reader's.
    static inline float value(device const Block & blk, uint j) {
        return (float)blk.qs[j] * (float)blk.d;
    }

    // Sixteen consecutive values of a block, [16*il, 16*il + 16) in natural
    // order (value 16*il + j at reg[j/4][j%4]), the GEMM's A staging: the
    // owner's own `value`, rounded once into the operand type.
    template <typename T>
    static inline void dequant16(device const Block & blk,
                                 short il,
                                 thread matrix<T, 4, 4> & reg) {
        const float d = (float)blk.d;
        device const int8_t * q = blk.qs + 16 * il;
        FOR_UNROLL(short j = 0; j < 16; ++j) {
            reg[j / 4][j % 4] = (float)q[j] * d;
        }
    }
};

// GGUF's Q5_K decode owner (block_q5_K, above): Q4_K's block with a fifth
// bit a value. A lane loads the same 32 values of a block Q4_K's lane loads
// (its quarter of a 64-value group pair), so the registers, the load and the
// group sums are Q4_K's own; the lane map is llama.cpp's
// kernel_mul_mv_q5_K_f32's, which walks four blocks apart as Q4_K's does
// but puts neighbouring lanes on neighbouring blocks. The dot is that
// kernel's: the nibbles under d and the fifth bits' sums times 16, less the
// dmin term.
struct Q5K {
    typedef block_q5_K Block;
    typedef Q4K::Lane Lane;

    static inline Lane lane_of(ushort tiisg) {
        Lane lane;
        lane.ix = (short)(tiisg % 4);  // 0...3
        const short tid = (short)(tiisg / 4);
        lane.iq = (short)(tid / 4);    // 0 or 1
        lane.ir = (short)(tid % 4);    // 0...3
        return lane;
    }

    static inline int first_block(const thread Lane &lane) {
        return lane.ix;
    }

    static inline int block_stride() {
        return 4;
    }

    static inline int values_per_block() {
        return QK_K;
    }

    static inline int input_offset(const thread Lane &lane) {
        return Q4K::input_offset(lane);
    }

    template <LoadMode LOAD>
    static inline void load(thread Lane &lane,
                            device const float * block,
                            device const float * gain,
                            float scale,
                            thread float * sum_sq) {
        Q4K::load<LOAD>(lane, block, gain, scale, sum_sq);
    }

    // The lane's four group dots of one block: acc1 the nibble sums (the
    // high nibbles 16 times over, as the byte masks leave them), acc2 the
    // sums of the activations whose fifth bit is set, and the half's scale
    // words.
    static inline void dots(device const Block & blk,
                            const thread Lane & lane,
                            thread float4 & acc1,
                            thread float4 & acc2,
                            thread uint16_t * sc16) {
        const short l0 = 8 * lane.ir;
        device const uint8_t * q1 = blk.qs + 32 * lane.iq + l0;
        device const uint8_t * q2 = q1 + 64;
        device const uint8_t * qh = blk.qh + l0;
        const uint8_t hm1 = 1u << (2 * lane.iq);
        const uint8_t hm2 = hm1 << 1;
        const uint8_t hm3 = hm1 << 4;
        const uint8_t hm4 = hm2 << 4;
        k4_scale_words((device const uint16_t *)blk.scales + lane.iq, sc16);
        acc1 = float4(0.f, 0.f, 0.f, 0.f);
        acc2 = float4(0.f, 0.f, 0.f, 0.f);
        FOR_UNROLL (short l = 0; l < 8; ++l) {
            const uint8_t h = qh[l];
            acc1[0] += lane.yl[l + 0] * (q1[l] & 0x0F);
            acc1[1] += lane.yl[l + 8] * (q1[l] & 0xF0);
            acc1[2] += lane.yh[l + 0] * (q2[l] & 0x0F);
            acc1[3] += lane.yh[l + 8] * (q2[l] & 0xF0);
            acc2[0] += h & hm1 ? lane.yl[l + 0] : 0.f;
            acc2[1] += h & hm2 ? lane.yl[l + 8] : 0.f;
            acc2[2] += h & hm3 ? lane.yh[l + 0] : 0.f;
            acc2[3] += h & hm4 ? lane.yh[l + 8] : 0.f;
        }
    }

    // One block's contribution, in llama.cpp's association: d times the
    // scaled code sums, less dmin times the scaled activation sums.
    static inline float partial(device const Block & blk,
                                const thread Lane & lane) {
        float4 acc1;
        float4 acc2;
        uint16_t sc16[4];
        dots(blk, lane, acc1, acc2, sc16);
        thread const uint8_t * sc8 = (thread const uint8_t *)sc16;
        const float d = (float)blk.d;
        const float dmin = (float)blk.dmin;
        return d * (sc8[0] * (acc1[0]        + 16.f * acc2[0]) +
                    sc8[1] * (acc1[1] / 16.f + 16.f * acc2[1]) +
                    sc8[4] * (acc1[2]        + 16.f * acc2[2]) +
                    sc8[5] * (acc1[3] / 16.f + 16.f * acc2[3]))
             - dmin * (lane.sumy[0] * sc8[2] + lane.sumy[1] * sc8[3] +
                       lane.sumy[2] * sc8[6] + lane.sumy[3] * sc8[7]);
    }

    // The same block dotted against a head-scaled input, Q4_K's two-halves
    // form: the block's two 128-value head groups, each scale on its whole
    // term once, the dmin subtraction per half.
    static inline float partial_head(device const Block & blk,
                                     const thread Lane & lane,
                                     float lo,
                                     float hi) {
        float4 acc1;
        float4 acc2;
        uint16_t sc16[4];
        dots(blk, lane, acc1, acc2, sc16);
        thread const uint8_t * sc8 = (thread const uint8_t *)sc16;
        const float d = (float)blk.d;
        const float dmin = (float)blk.dmin;
        return lo * (d * (sc8[0] * (acc1[0]        + 16.f * acc2[0]) +
                          sc8[1] * (acc1[1] / 16.f + 16.f * acc2[1])) -
                     dmin * (lane.sumy[0] * sc8[2] + lane.sumy[1] * sc8[3])) +
               hi * (d * (sc8[4] * (acc1[2]        + 16.f * acc2[2]) +
                          sc8[5] * (acc1[3] / 16.f + 16.f * acc2[3])) -
                     dmin * (lane.sumy[2] * sc8[6] + lane.sumy[3] * sc8[7]));
    }

    static inline float2 head_scales(threadgroup const float * scales, int ib) {
        return Q4K::head_scales(scales, ib);
    }

    static inline bool sums_finite(const thread Lane &lane) {
        return Q4K::sums_finite(lane);
    }

    static inline int fold_blocks() {
        return BLOOMERY_EXPERT_SLOTS * (BLOOMERY_EXPERT_K / values_per_block());
    }

    static inline int fold_slot(int ib) {
        return ib / (BLOOMERY_EXPERT_K / values_per_block());
    }

    // The scale and the min of 32-value group `s` (0...7) times d and dmin:
    // the reader's d1 and m1, both exact in f32.
    static inline float2 group_scale(device const Block & blk, short s) {
        uint16_t sc16[4];
        thread const uint8_t * sc8 = (thread const uint8_t *)sc16;
        k4_scale_words((device const uint16_t *)blk.scales + ((s >> 1) & 1), sc16);
        const short at = (s & 1) + 4 * (s >> 2);
        return float2((float)blk.d * (float)sc8[at], (float)blk.dmin * (float)sc8[2 + at]);
    }

    // The five-bit code of value `j`: its nibble (the low one in the first
    // 32 values of a 64-value group, the high one in the second) and its
    // fifth bit.
    static inline uint code(device const Block & blk, uint j) {
        const uint group = j / 64;
        const uint high = (j / 32) & 1;
        const uint l = j % 32;
        const uint nibble = (blk.qs[32 * group + l] >> (4 * high)) & 0x0F;
        const uint fifth = (blk.qh[l] >> (2 * group + high)) & 1;
        return nibble | (fifth << 4);
    }

    // Value `j` of a block, the host reader's decode: d1 * q - m1, where
    // d1 * q (at most 22 significant bits) and m1 are exact in f32, so the
    // subtraction is the one rounding and the embed's bits are the reader's.
    static inline float value(device const Block & blk, uint j) {
        const float2 sm = group_scale(blk, (short)(j / 32));
        return sm.x * (float)code(blk, j) - sm.y;
    }

    // Sixteen consecutive values of a block, [16*il, 16*il + 16) in natural
    // order (value 16*il + j at reg[j/4][j%4]), the GEMM's A staging: the
    // owner's own value with the group's scale and min taken once (a chunk
    // sits inside one 32-value group), computed in f32 and rounded once
    // into the operand type.
    template <typename T>
    static inline void dequant16(device const Block & blk,
                                 short il,
                                 thread matrix<T, 4, 4> & reg) {
        const float2 sm = group_scale(blk, il >> 1);
        FOR_UNROLL(short j = 0; j < 16; ++j) {
            reg[j / 4][j % 4] = sm.x * (float)code(blk, (uint)(16 * il + j)) - sm.y;
        }
    }
};

// Sixteen f32 values, 64 bytes: the block f32 weight rows are read in. The
// router's GEMM (`F32W`, gemm_routed.metal) stages it sixteen values at a
// time; the segmented mat-vec's `F32` owner below splits it over four lanes.
struct block_f32_16 {
    float v[16];
};

static_assert(sizeof(block_f32_16) == 64, "block_f32_16 is 64 bytes");

// The f32 weight rows' decode owner (block_f32_16, above): the values as
// they are, `y = w · x`. Four lanes split a 16-value block into quarters of
// four values, one float4 each, so the 32 lanes cover eight blocks, 128
// values, a pass (the stride). The owner serves the segmented mat-vec, the
// one launch that reads f32 weights as a layer's projection (the delta net's
// `ssm_alpha` and `ssm_beta`); the bodies that need `fold_blocks`, `value` or
// `dequant16` are not instantiated over it. A quarter's dot is two pairs of
// products added, so the sum order is fixed here, not by the compiler.
struct F32 {
    typedef block_f32_16 Block;

    // The lane's map and registers: ix picks the lane's blocks (eight
    // apart), il the quarter of one. yl holds the loaded activations, four
    // consecutive values.
    struct Lane {
        short   ix;
        short   il;
        float   yl[4];
    };

    static inline Lane lane_of(ushort tiisg) {
        Lane lane;
        lane.ix = (short)(tiisg / 4);  // 0...7
        lane.il = (short)(tiisg % 4);  // 0...3
        return lane;
    }

    static inline int first_block(const thread Lane &lane) {
        return lane.ix;
    }

    static inline int block_stride() {
        return 8;
    }

    static inline int values_per_block() {
        return 16;
    }

    // The lane's values' offset in a block: its quarter.
    static inline int input_offset(const thread Lane &lane) {
        return 4 * lane.il;
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
        float a0 = block[base + 0];
        float a1 = block[base + 1];
        float a2 = block[base + 2];
        float a3 = block[base + 3];
        if (LOAD == LOAD_ROUTE) {
            a0 *= scale;
            a1 *= scale;
            a2 *= scale;
            a3 *= scale;
        } else if (LOAD == LOAD_NORM) {
            *sum_sq += (a0 * a0 + a1 * a1) + (a2 * a2 + a3 * a3);
            a0 *= gain[base + 0];
            a1 *= gain[base + 1];
            a2 *= gain[base + 2];
            a3 *= gain[base + 3];
        }
        lane.yl[0] = a0;
        lane.yl[1] = a1;
        lane.yl[2] = a2;
        lane.yl[3] = a3;
    }

    // One block's contribution: the lane's quarter of the weights against
    // its registers, two pairs of products added.
    static inline float partial(device const Block & blk,
                                const thread Lane & lane) {
        const float4 w = *(device const float4 *)(blk.v + 4 * lane.il);
        return (w.x * lane.yl[0] + w.y * lane.yl[1]) + (w.z * lane.yl[2] + w.w * lane.yl[3]);
    }
};
