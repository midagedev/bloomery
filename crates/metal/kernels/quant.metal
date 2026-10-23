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
//                count) — QK_K for Q4_K and Q6_K, 64 for MLX's groups
//   load<LOAD>   its input load — plain, the norm on load (the gain and the
//                sum of squares), or the route-weight form
//   partial      the dot of one block against the loaded input
//   fold_blocks, the folded down's K' in the quant's blocks, and the slot
//   fold_slot    a block of K' belongs to; `head_scales` and `sums_finite`
//                serve the head-normed row loop, and `value` the embedding
//
// Every kernel body over quantized weights — the mat-vec's row loop with its
// modes, the expert pair's SwiGLU, the folded down, the head's argmax, the
// embedding — is a template over this; the entry points are the
// instantiations, under their host names. A new weight format adds one owner
// here and instantiates the bodies; no body is copied per quant.
//
// The lane maps (Q4_K's ix/iq/ir, Q6_K's tid/ix/ip/il), the block strides,
// the uint16 nibble trick, the 6-bit unpacking and the sum associations are
// llama.cpp's, from its kernel_mul_mv_q4_K_f32 and kernel_mul_mv_q6_K_f32
// (MIT licence; see THIRD_PARTY_NOTICES.md):
//   ggml/src/ggml-metal/kernels/mul_mv.metal
// The same map was the dense mat-vec's, the pair's and the fold's since each
// landed; this file is those copies folded into one. A4's map is ours, on
// the same idiom (the nibble trick above), over MLX's group layout.
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

// MLX's affine 4-bit group (crates/gguf's MLX_A4): 64 values, 36 bytes —
// eight u32 words of codes, then a BF16 scale and a BF16 bias. Value 8w + j
// is the 4 bits at bit 4j of word w, the low nibble first.
struct block_mlx_a4 {
    uint32_t qs[8];
    uint16_t scale;  // BF16 bits
    uint16_t bias;   // BF16 bits
};

static_assert(sizeof(block_mlx_a4) == 36, "block_mlx_a4 is 36 bytes");

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
        for (short i = 0; i < 8; ++i) {
            float a0 = block[base + i +  0];
            float a1 = block[base + i + 32];
            float a2 = block[base + i + 128];
            float a3 = block[base + i + 160];
            if (LOAD == LOAD_ROUTE) {
                a0 *= scale;
                a1 *= scale;
                a2 *= scale;
                a3 *= scale;
            } else if (LOAD == LOAD_NORM) {
                *sum_sq += (a0 * a0 + a1 * a1) + (a2 * a2 + a3 * a3);
                a0 *= gain[base + i +  0];
                a1 *= gain[base + i + 32];
                a2 *= gain[base + i + 128];
                a3 *= gain[base + i + 160];
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

    // One block's contribution. The subtraction keeps llama.cpp's
    // association: d times the nibble sum, minus dmin times the activation
    // sum.
    static inline float partial(device const Block & blk,
                                const thread Lane & lane) {
        constexpr uint16_t kmask1 = 0x3f3f;
        constexpr uint16_t kmask2 = 0x0f0f;
        constexpr uint16_t kmask3 = 0xc0c0;

        uint16_t sc16[4];
        thread const uint8_t * sc8 = (thread const uint8_t *)sc16;
        device const uint16_t * sc = (device const uint16_t *)blk.scales + lane.iq;
        device const uint16_t * q1 = (device const uint16_t *)blk.qs + 16 * lane.iq + 4 * lane.ir;
        device const half * dh = &blk.d;

        sc16[0] = sc[0] & kmask1;
        sc16[1] = sc[2] & kmask1;
        sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
        sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

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
        constexpr uint16_t kmask1 = 0x3f3f;
        constexpr uint16_t kmask2 = 0x0f0f;
        constexpr uint16_t kmask3 = 0xc0c0;

        uint16_t sc16[4];
        thread const uint8_t * sc8 = (thread const uint8_t *)sc16;
        device const uint16_t * sc = (device const uint16_t *)blk.scales + lane.iq;
        device const uint16_t * q1 = (device const uint16_t *)blk.qs + 16 * lane.iq + 4 * lane.ir;
        device const half * dh = &blk.d;

        sc16[0] = sc[0] & kmask1;
        sc16[1] = sc[2] & kmask1;
        sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
        sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

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
        for (short l = 0; l < 4; ++l) {
            float a0 = block[base + l +  0];
            float a1 = block[base + l + 32];
            float a2 = block[base + l + 64];
            float a3 = block[base + l + 96];
            if (LOAD == LOAD_ROUTE) {
                a0 *= scale;
                a1 *= scale;
                a2 *= scale;
                a3 *= scale;
            } else if (LOAD == LOAD_NORM) {
                *sum_sq += (a0 * a0 + a1 * a1) + (a2 * a2 + a3 * a3);
                a0 *= gain[base + l +  0];
                a1 *= gain[base + l + 32];
                a2 *= gain[base + l + 64];
                a3 *= gain[base + l + 96];
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
};

// MLX's affine 4-bit decode owner (block_mlx_a4, above): one 64-value group
// is one block, so a block count a body reads is a group count and the lane
// map splits one group across four lanes — a SIMD group's 32 lanes cover
// eight groups, 512 values, a pass (the stride). The lane's two consecutive
// code words hold its sixteen values in address order (nibble j of a word is
// value j of the word), so the uint16 mask trick reads each nibble of a word
// pair into its own accumulator and the shift of a whole accumulator folds
// once into an exact power of two. MLX's own `affine_qmv_fast` (quantized.h)
// is the design reference for the bias fold; its four-rows-a-thread x reuse
// is this engine's SIMD-group row loop instead.
struct A4 {
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
        for (short l = 0; l < 4; ++l) {
            float a0 = block[base + 4 * l + 0];
            float a1 = block[base + 4 * l + 1];
            float a2 = block[base + 4 * l + 2];
            float a3 = block[base + 4 * l + 3];
            if (LOAD == LOAD_ROUTE) {
                a0 *= scale;
                a1 *= scale;
                a2 *= scale;
                a3 *= scale;
            } else if (LOAD == LOAD_NORM) {
                *sum_sq += (a0 * a0 + a1 * a1) + (a2 * a2 + a3 * a3);
                a0 *= gain[base + 4 * l + 0];
                a1 *= gain[base + 4 * l + 1];
                a2 *= gain[base + 4 * l + 2];
                a3 *= gain[base + 4 * l + 3];
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

    // The group's BF16 pair, widened by the shift: exact.
    static inline float scale_of(device const Block & blk) {
        return as_type<float>((uint)blk.scale << 16);
    }

    static inline float bias_of(device const Block & blk) {
        return as_type<float>((uint)blk.bias << 16);
    }

    // One group's contribution. The scale multiplies the code dot and the
    // bias the lane's input sum, as MLX's `qdot` folds its bias and Q4_K its
    // dmin term.
    static inline float partial(device const Block & blk,
                                const thread Lane & lane) {
        device const uint16_t * q = (device const uint16_t *)blk.qs + 4 * lane.iq;
        float4 acc = {0.f, 0.f, 0.f, 0.f};
        FOR_UNROLL (short m = 0; m < 4; ++m) {
            acc[0] += lane.yl[4 * m + 0] * (q[m] & 0x000F);
            acc[1] += lane.yl[4 * m + 1] * (q[m] & 0x00F0);
            acc[2] += lane.yl[4 * m + 2] * (q[m] & 0x0F00);
            acc[3] += lane.yl[4 * m + 3] * (q[m] & 0xF000);
        }
        return scale_of(blk) * ((acc[0] + 1.f/16.f * acc[1]) +
                                (1.f/256.f * acc[2] + 1.f/4096.f * acc[3]))
             + bias_of(blk) * lane.sumy;
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
    // 16-bit words: a group's 36 bytes stride every block to the same mod 4,
    // so a wider read would ask an alignment no launcher checks.
    static inline float value(device const Block & blk, uint j) {
        device const uint16_t * w = (device const uint16_t *)blk.qs;
        const uint word = (uint)w[2 * (j / 8)] | ((uint)w[2 * (j / 8) + 1] << 16);
        const float q = float((word >> (4 * (j % 8))) & 0xF);
        return scale_of(blk) * q + bias_of(blk);
    }
};
