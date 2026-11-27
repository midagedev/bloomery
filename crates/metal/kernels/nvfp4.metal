// The NVFP4 mat-vec of Inkling 2's routed experts over the fused stacks the
// family's files hold their first twenty stored layers' experts in — the
// GGUF's blocks a SIMD group a row, the MLX folder's planes sixteen rows a
// SIMD group — the per-expert global scale applied after the dot.
//
// The block (mainline's block_nvfp4, 64 values, 36 bytes): bytes 0..4 the
// UE4M3 scales of the four 16-value sub-blocks, bytes 4..36 the codes.
// Sub-block s's value j < 8 is the low nibble of code byte 8s + j and value
// j + 8 its high nibble; a code c means KVALUES[c] · ue4m3(d[s]) / 2 — the
// E2M1 significand table and the scale both exact in f32, so every decoded
// value is exact and the dot's products are the only rounding (the card
// side dots the same codes against quantized activations with exact integer
// sub-block sums; this backend's activations are f32, and the accumulation
// is f32 fused multiply-adds — a named band term between the two).
//
// The lane map: a SIMD group owns a row, lane l the code byte l of every
// block (a block holds 32 code bytes, so the 32 lanes cover it exactly), the
// blocks walked ascending. Byte b of a block holds sub-block b/8's values
// b%8 (the low nibble) and b%8 + 8 (the high), so a lane's sub-block is
// lane-uniform in eights and its two activations load as two scalars eight
// apart. A lane accumulates its two products a block by fused-free adds in
// block order; the group's lanes are summed by simd_sum.
//
// The per-expert global scale (the file's `<stack>.scale` rows, one f32 an
// expert) multiplies each row's finished dot once — the same elementwise
// product either way, exact, so its placement costs nothing.
//
// The route: a launch covers every slot of a route at once, the grid's
// height a slot. A slot reads its expert's id on the card through the
// experts' common id read (pair_expert, experts.metal), its expert's rows
// at the id's offset in the stack, its expert's scale out of the stack's
// scale row and, on a routed down, its route weight — the scale of its
// rows' finished dots the f32 product scale · weight. Its input is the one
// row every slot shares (a gate·up) or its own (a down), and its outputs
// land in planes of `plane` rows: row r of slot s at
// ((r / plane) · slots + s) · plane + r % plane, so a gate·up's gate rows of
// every slot come first and its up rows after them.
//
// No silent failure: a scale byte that is no UE4M3 (bit 7 set, or E4M3's
// NaN 0x7f) has no defined scale; the decode is NaN, the quant-column site
// raises and the row is NaN — never a plausible dot. A row whose dot stops
// being finite raises the product site. A slot whose id names no expert of
// the stack, or a slot past the row's routed ids, raises the expert-id site
// and its rows are NaN.
//
// This file comes after fault.metal (raise) and experts.metal (pair_expert).

#pragma METAL fp contract(off)

/// Values one NVFP4 block holds: four 16-value sub-blocks.
#define NV_VALUES 64u
/// Code bytes one block holds: two values a byte.
#define NV_CODES 32u

/// The block: the four sub-blocks' UE4M3 scale bytes as one word, then the
/// codes.
struct block_nvfp4 {
    uint32_t e;
    uint8_t qs[NV_CODES];
};

static_assert(sizeof(block_nvfp4) == 36, "block_nvfp4 is 36 bytes");

// The E2M1 significand table, doubled: a code's mantissa value.
static constant float NV_KVALUES[16] = {
    0.f, 1.f, 2.f, 3.f, 4.f, 6.f, 8.f, 12.f,
    0.f, -1.f, -2.f, -3.f, -4.f, -6.f, -8.f, -12.f,
};

// `ue4m3(b) / 2` for a UE4M3 scale byte, NaN for a byte that is none (bit 7
// set, or E4M3's NaN): both halves of the decode exact in f32.
inline float nvfp4_scale_half(uint b) {
    if ((b & 0x80u) != 0u || b == 0x7fu) {
        return NAN;
    }
    const uint exp = b >> 3;
    const float man = float(b & 7u);
    const float raw = exp == 0u ? man * exp2(-9.0f) : (1.0f + man / 8.0f) * exp2(float(exp) - 7.0f);
    return raw * 0.5f;
}

// The launch's argument block, bound by value.
struct Nvfp4Args {
    uint k;        // the input's values, a multiple of the block's
    uint rows;     // an expert's rows, every one of them a slot's launch
    uint layer;
    uint experts;  // the stack's experts: an id at or past it names none
    uint slots;    // the route's slots, the grid's height
    uint plane;    // rows an output plane holds; it divides rows
    uint x_stride; // values between two slots' inputs, 0 for one shared row
    uint weighted; // 1: a slot's scale is its expert's times its route weight
};

static_assert(sizeof(Nvfp4Args) == 32, "Nvfp4Args is 32 bytes: the host packs eight words");

// Slot `slot`'s expert and the scale of its rows' finished dots, or false
// when the slot names no expert: an id past the stack (pair_expert's
// refusal), or a slot past the row's routed ids, which no id reads.
inline bool nv_route(device const int *ids,
                     device const float *scales,
                     device const float *route_w,
                     constant Nvfp4Args &args,
                     uint slot,
                     thread ulong &expert,
                     thread float &w_scale)
{
    int e = 0;
    if (slot >= BLOOMERY_EXPERT_ROUTED || pair_expert(ids, slot, args.experts, e)) {
        return false;
    }
    expert = ulong(e);
    w_scale = args.weighted != 0u ? scales[e] * route_w[slot] : scales[e];
    return true;
}

// Where row `r` of slot `slot` lands: its plane's slot block.
inline uint nv_out(constant Nvfp4Args &args, uint slot, uint r)
{
    return ((r / args.plane) * args.slots + slot) * args.plane + r % args.plane;
}

// The rows of a slot that names no expert: NaN, the expert-id site raised.
inline void nv_bad_slot(device float *y, constant Nvfp4Args &args, device atomic_uint *faults, uint out)
{
    raise(faults, args.layer, BLOOMERY_FAULT_SITE_EXPERT_ID);
    y[out] = NAN;
}

kernel void nvfp4_gemv(device const block_nvfp4 *W [[buffer(0)]],
                       device const float *x [[buffer(1)]],
                       device float *y [[buffer(2)]],
                       constant Nvfp4Args &args [[buffer(3)]],
                       device atomic_uint *faults [[buffer(4)]],
                       device const int *ids [[buffer(5)]],
                       device const float *scales [[buffer(6)]],
                       device const float *route_w [[buffer(7)]],
                       uint2 tg [[threadgroup_position_in_grid]],
                       uint lane [[thread_index_in_simdgroup]],
                       uint sg [[simdgroup_index_in_threadgroup]])
{
    const uint r = tg.x * 8u + sg;
    const uint slot = tg.y;
    if (r >= args.rows || slot >= args.slots) {
        return;
    }
    const uint out = nv_out(args, slot, r);
    ulong expert = 0;
    float w_scale = 0.f;
    if (!nv_route(ids, scales, route_w, args, slot, expert, w_scale)) {
        if (lane == 0u) {
            nv_bad_slot(y, args, faults, out);
        }
        return;
    }
    x += slot * args.x_stride;
    const uint blocks = args.k / NV_VALUES;
    device const block_nvfp4 *wr = W + (expert * (ulong)args.rows + (ulong)r) * (ulong)blocks;
    // The lane's sub-block: byte b of a block is sub-block b/8's pair, and
    // the lane's byte of every block is its own l, so a lane's sub-block is
    // fixed by its lane index in eights.
    const uint sub = lane / 8u;
    const uint j = lane % 8u;
    bool usable = true;
    float acc = 0.f;
    for (uint i = 0u; i < blocks; ++i) {
        const device block_nvfp4 &blk = wr[i];
        const float s = nvfp4_scale_half((blk.e >> (8u * sub)) & 0xffu);
        usable = usable && isfinite(s);
        const uint c0 = blk.qs[lane] & 0x0fu;
        const uint c1 = blk.qs[lane] >> 4u;
        const float v0 = NV_KVALUES[c0] * s;
        const float v1 = NV_KVALUES[c1] * s;
        const uint col = i * NV_VALUES + sub * 16u + j;
        acc += v0 * x[col];
        acc += v1 * x[col + 8u];
    }
    const float dot = simd_sum(acc) * w_scale;
    // One lane's scale byte that is none makes the whole row undefined.
    usable = simd_all(usable);
    if (!usable) {
        if (lane == 0u) {
            raise(faults, args.layer, BLOOMERY_FAULT_SITE_QUANT_COLUMN);
            y[out] = NAN;
        }
        return;
    }
    if (!isfinite(dot) && lane == 0u) {
        raise(faults, args.layer, BLOOMERY_FAULT_SITE_F32_PRODUCT);
    }
    if (lane == 0u) {
        y[out] = isfinite(dot) ? dot : NAN;
    }
}

// The planes entry (nvfp4_planes_gemv) is bloomery's NVFP4 planes layout, the
// MLX folder's two planes an expert with the rows in the launch's order: the
// codes plane (k / 2 bytes a row, value 2i the low nibble of byte i), then
// the scales plane (k / 16 UE4M3 bytes a row, one a 16-value group). Expert
// e's planes start at e · rows · k · 9 / 16 bytes.
//
// Its lane map: a SIMD group owns 16 rows as four sets of four, set q the
// lanes 8q .. 8q + 7 and the rows 4q .. 4q + 3 of the group's 16; lane j of a
// set takes the groups j, j + 8, ... of its four rows. A step loads the
// lane's 16 inputs once (four float4) and, against them, each of its rows'
// group: one 8-byte code word and one scale byte. A row past the launch's
// reads the last row and stores nothing.
//
// Its decode is arithmetic: an E2M1 code's magnitude bits placed at f16 bits
// 9 .. 11 and its sign at bit 15 make the f16 d(c) · 2^-15 (d the doubled
// table above; codes 0 and 1 subnormal, exact), two codes at once from a
// code word's halves; a UE4M3 byte placed at f16 bits 7 .. 14 makes
// ue4m3 · 2^-8. A value is d(c) · ue4m3 / 2, so a group's scale is that f16
// times 2^22, exact. A scale byte at or past 0x7f is none (NaN or bit 7).
//
// This order is the gate: a group's 16 products (a code's f16 times its
// input) sum as four fused chains, chain p the group's values p, p + 4,
// p + 8, p + 12 in that order, and the chains (0 + 1) + (2 + 3); the group's
// sum times its scale is fused into the lane's row sum, the groups
// ascending; a set's eight lanes are summed by the butterfly at 4, 2, 1; the
// slot's scale multiplies the finished dot.

/// Rows a planes set holds: the rows a lane's inputs serve.
#define NVP_ROWS 4u
/// Lanes a planes set holds: a row's groups are dealt over them.
#define NVP_LANES 8u
/// Rows a planes SIMD group owns: its four sets'.
#define NVP_SG_ROWS (NVP_ROWS * (32u / NVP_LANES))
/// SIMD groups a planes threadgroup holds.
#define NVP_SGS 2u
/// The factor that makes the f16 scale's ue4m3 · 2^-8 a value's
/// ue4m3 / 2 against the code's d(c) · 2^-15: 2^22.
#define NVP_SCALE 4194304.0f

// The codes p and p + 4 of the code word `w` (values 0 .. 7 of a group's
// half) as f16 d(c) · 2^-15, widened.
template <uint P>
inline float2 nvp_pair(uint w) {
    const uint mag = P < 3u ? (w << (9u - 4u * P)) : (w >> 3u);
    const uint sgn = P < 3u ? (w << (12u - 4u * P)) : w;
    return float2(as_type<half2>((mag & 0x0E000E00u) | (sgn & 0x80008000u)));
}

// A group's sum of its 16 products: the codes `c` (values 0 .. 7 in c.x,
// 8 .. 15 in c.y) against the inputs `xv` (values 4i .. 4i + 3 in xv[i]),
// in the order above.
inline float nvp_group(uint2 c, thread const float4 *xv) {
    const float2 a0 = nvp_pair<0u>(c.x);
    const float2 a1 = nvp_pair<1u>(c.x);
    const float2 a2 = nvp_pair<2u>(c.x);
    const float2 a3 = nvp_pair<3u>(c.x);
    const float2 b0 = nvp_pair<0u>(c.y);
    const float2 b1 = nvp_pair<1u>(c.y);
    const float2 b2 = nvp_pair<2u>(c.y);
    const float2 b3 = nvp_pair<3u>(c.y);
    float s0 = a0.x * xv[0].x;
    float s1 = a1.x * xv[0].y;
    float s2 = a2.x * xv[0].z;
    float s3 = a3.x * xv[0].w;
    s0 = fma(a0.y, xv[1].x, s0);
    s1 = fma(a1.y, xv[1].y, s1);
    s2 = fma(a2.y, xv[1].z, s2);
    s3 = fma(a3.y, xv[1].w, s3);
    s0 = fma(b0.x, xv[2].x, s0);
    s1 = fma(b1.x, xv[2].y, s1);
    s2 = fma(b2.x, xv[2].z, s2);
    s3 = fma(b3.x, xv[2].w, s3);
    s0 = fma(b0.y, xv[3].x, s0);
    s1 = fma(b1.y, xv[3].y, s1);
    s2 = fma(b2.y, xv[3].z, s2);
    s3 = fma(b3.y, xv[3].w, s3);
    return (s0 + s1) + (s2 + s3);
}

kernel void nvfp4_planes_gemv(device const uint8_t *W [[buffer(0)]],
                              device const float *x [[buffer(1)]],
                              device float *y [[buffer(2)]],
                              constant Nvfp4Args &args [[buffer(3)]],
                              device atomic_uint *faults [[buffer(4)]],
                              device const int *ids [[buffer(5)]],
                              device const float *scales [[buffer(6)]],
                              device const float *route_w [[buffer(7)]],
                              uint2 tg [[threadgroup_position_in_grid]],
                              uint lane [[thread_index_in_simdgroup]],
                              uint sg [[simdgroup_index_in_threadgroup]])
{
    const uint r0 = (tg.x * NVP_SGS + sg) * NVP_SG_ROWS;
    const uint slot = tg.y;
    if (r0 >= args.rows || slot >= args.slots) {
        return;
    }
    ulong expert = 0;
    float w_scale = 0.f;
    if (!nv_route(ids, scales, route_w, args, slot, expert, w_scale)) {
        if (lane < NVP_SG_ROWS && r0 + lane < args.rows) {
            nv_bad_slot(y, args, faults, nv_out(args, slot, r0 + lane));
        }
        return;
    }
    x += slot * args.x_stride;
    const uint set = lane / NVP_LANES;
    const uint j = lane % NVP_LANES;
    const uint rs = r0 + set * NVP_ROWS;
    const uint groups = args.k / 16u;
    const ulong code_row = (ulong)groups * 8ul;
    device const uint8_t *base = W + expert * (ulong)args.rows * (ulong)groups * 9ul;
    device const uint8_t *sbase = base + (ulong)args.rows * code_row;
    device const uint2 *cr[NVP_ROWS];
    device const uint8_t *sr[NVP_ROWS];
    for (uint i = 0u; i < NVP_ROWS; ++i) {
        const ulong r = (ulong)min(rs + i, args.rows - 1u);
        cr[i] = (device const uint2 *)(base + r * code_row);
        sr[i] = sbase + r * (ulong)groups;
    }
    device const float4 *xq = (device const float4 *)x;
    float acc[NVP_ROWS] = {0.f, 0.f, 0.f, 0.f};
    uint bad = 0u;
    for (uint g = j; g < groups; g += NVP_LANES) {
        float4 xv[4];
        for (uint i = 0u; i < 4u; ++i) {
            xv[i] = xq[4u * g + i];
        }
        for (uint i = 0u; i < NVP_ROWS; ++i) {
            const uint2 c = cr[i][g];
            const uint b = uint(sr[i][g]);
            bad |= (b >= 0x7fu ? 1u : 0u) << i;
            const float s = float(as_type<half>(ushort(b << 7u))) * NVP_SCALE;
            acc[i] = fma(nvp_group(c, xv), s, acc[i]);
        }
    }
    for (uint i = 0u; i < NVP_ROWS; ++i) {
        acc[i] += simd_shuffle_xor(acc[i], 4u);
        acc[i] += simd_shuffle_xor(acc[i], 2u);
        acc[i] += simd_shuffle_xor(acc[i], 1u);
    }
    bad |= simd_shuffle_xor(bad, 4u);
    bad |= simd_shuffle_xor(bad, 2u);
    bad |= simd_shuffle_xor(bad, 1u);
    // Lane j < 4 of a set stores the set's row j.
    const uint r = rs + j;
    if (j >= NVP_ROWS || r >= args.rows) {
        return;
    }
    const float sum = j == 0u ? acc[0] : j == 1u ? acc[1] : j == 2u ? acc[2] : acc[3];
    const uint out = nv_out(args, slot, r);
    if (((bad >> j) & 1u) != 0u) {
        raise(faults, args.layer, BLOOMERY_FAULT_SITE_QUANT_COLUMN);
        y[out] = NAN;
        return;
    }
    const float dot = sum * w_scale;
    if (!isfinite(dot)) {
        raise(faults, args.layer, BLOOMERY_FAULT_SITE_F32_PRODUCT);
    }
    y[out] = isfinite(dot) ? dot : NAN;
}

// The two other forms a routed stack of the family's files takes, beside the
// NVFP4 ones, each with the same route: the GGUF coda's Q8_0 blocks
// (q8_0_scaled_gemv) and the MLX folder coda's MXFP8 rows
// (mxfp8_rows_gemv). The coda's experts have no global scale (their scale
// row is ones), so a routed down's scale is its route weight and a routed
// down of either form leaves its expert's weighted output as the NVFP4 one
// does.
//
// Q8_0 (34 bytes a 32-value block: the f16 scale, then 32 int8 codes): a code
// q means q · d, exact in f32 (an int8 times an f16). A SIMD group a row, lane
// l the code l of every block, the blocks ascending. A scale that is not
// finite has no defined value: the row is NaN and the quant-column site
// raises.
//
// MXFP8 rows (33 bytes a 32-value group: the E8M0 scale byte, then 32 E4M3
// codes, value i code byte i): a code means e4m3(c) · 2^(s − 127), exact in
// f32. Scale byte 0xff and the E4M3 NaN codes (0x7f, 0xff) have no defined
// value: the row is NaN and the quant-column site raises. Lane l the code l
// of every group, the groups ascending.

/// The f32 value of the f16 bits `lo | hi << 8`.
inline float q8_0_scale(uint lo, uint hi) {
    return float(as_type<half>(ushort(lo | (hi << 8u))));
}

kernel void q8_0_scaled_gemv(device const uint8_t *W [[buffer(0)]],
                             device const float *x [[buffer(1)]],
                             device float *y [[buffer(2)]],
                             constant Nvfp4Args &args [[buffer(3)]],
                             device atomic_uint *faults [[buffer(4)]],
                             device const int *ids [[buffer(5)]],
                             device const float *scales [[buffer(6)]],
                             device const float *route_w [[buffer(7)]],
                             uint2 tg [[threadgroup_position_in_grid]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint sg [[simdgroup_index_in_threadgroup]])
{
    const uint r = tg.x * 8u + sg;
    const uint slot = tg.y;
    if (r >= args.rows || slot >= args.slots) {
        return;
    }
    const uint out = nv_out(args, slot, r);
    ulong expert = 0;
    float w_scale = 0.f;
    if (!nv_route(ids, scales, route_w, args, slot, expert, w_scale)) {
        if (lane == 0u) {
            nv_bad_slot(y, args, faults, out);
        }
        return;
    }
    x += slot * args.x_stride;
    const uint blocks = args.k / 32u;
    device const uint8_t *wr = W + (expert * (ulong)args.rows + (ulong)r) * (ulong)blocks * 34ul;
    bool usable = true;
    float acc = 0.f;
    for (uint i = 0u; i < blocks; ++i) {
        device const uint8_t *blk = wr + (ulong)i * 34ul;
        const float d = q8_0_scale(uint(blk[0]), uint(blk[1]));
        usable = usable && isfinite(d);
        const float v = float(as_type<char>(blk[2u + lane])) * d;
        acc += v * x[i * 32u + lane];
    }
    const float dot = simd_sum(acc) * w_scale;
    usable = simd_all(usable);
    if (!usable) {
        if (lane == 0u) {
            raise(faults, args.layer, BLOOMERY_FAULT_SITE_QUANT_COLUMN);
            y[out] = NAN;
        }
        return;
    }
    if (!isfinite(dot) && lane == 0u) {
        raise(faults, args.layer, BLOOMERY_FAULT_SITE_F32_PRODUCT);
    }
    if (lane == 0u) {
        y[out] = isfinite(dot) ? dot : NAN;
    }
}

/// The value of E4M3 code `c`, NaN for the two NaN codes.
inline float mxfp8_e4m3(uint c) {
    const uint mag = c & 0x7fu;
    if (mag == 0x7fu) {
        return NAN;
    }
    const uint exp = mag >> 3;
    const float man = float(mag & 7u);
    const float v = exp == 0u ? man * exp2(-9.0f) : (1.0f + man / 8.0f) * exp2(float(exp) - 7.0f);
    return (c & 0x80u) != 0u ? -v : v;
}

kernel void mxfp8_rows_gemv(device const uint8_t *W [[buffer(0)]],
                            device const float *x [[buffer(1)]],
                            device float *y [[buffer(2)]],
                            constant Nvfp4Args &args [[buffer(3)]],
                            device atomic_uint *faults [[buffer(4)]],
                            device const int *ids [[buffer(5)]],
                            device const float *scales [[buffer(6)]],
                            device const float *route_w [[buffer(7)]],
                            uint2 tg [[threadgroup_position_in_grid]],
                            uint lane [[thread_index_in_simdgroup]],
                            uint sg [[simdgroup_index_in_threadgroup]])
{
    const uint r = tg.x * 8u + sg;
    const uint slot = tg.y;
    if (r >= args.rows || slot >= args.slots) {
        return;
    }
    const uint out = nv_out(args, slot, r);
    ulong expert = 0;
    float w_scale = 0.f;
    if (!nv_route(ids, scales, route_w, args, slot, expert, w_scale)) {
        if (lane == 0u) {
            nv_bad_slot(y, args, faults, out);
        }
        return;
    }
    x += slot * args.x_stride;
    const uint groups = args.k / 32u;
    device const uint8_t *wr = W + (expert * (ulong)args.rows + (ulong)r) * (ulong)groups * 33ul;
    bool usable = true;
    float acc = 0.f;
    for (uint g = 0u; g < groups; ++g) {
        device const uint8_t *grp = wr + (ulong)g * 33ul;
        const uint sb = uint(grp[0]);
        const float s = sb == 0xffu ? NAN : exp2(float(sb) - 127.0f);
        const float v = mxfp8_e4m3(uint(grp[1u + lane])) * s;
        usable = usable && isfinite(v);
        acc += v * x[g * 32u + lane];
    }
    const float dot = simd_sum(acc) * w_scale;
    usable = simd_all(usable);
    if (!usable) {
        if (lane == 0u) {
            raise(faults, args.layer, BLOOMERY_FAULT_SITE_QUANT_COLUMN);
            y[out] = NAN;
        }
        return;
    }
    if (!isfinite(dot) && lane == 0u) {
        raise(faults, args.layer, BLOOMERY_FAULT_SITE_F32_PRODUCT);
    }
    if (lane == 0u) {
        y[out] = isfinite(dot) ? dot : NAN;
    }
}

// MXFP8 quads (132 bytes a 128-value block of four groups): the block's 128
// code bytes first, lane l's four codes (groups 4b to 4b + 3) the word at byte
// 4l, then the four groups' E8M0 scale bytes. Same bytes as the rows layout,
// the same value rule, the same per-lane order (group 0, 1, 2, ...): a SIMD
// group loads 128 bytes contiguous for four groups, and the decode is integer.

// E4M3 code byte `gg` of word `cw` times the E8M0 scale byte `gg` of word `sw`.
// `mag << 7` is the code's f16 pattern times 2^-8 (a subnormal code a subnormal
// half), and the scale is the f32 whose exponent field is the byte (byte 0 is
// 2^-127, the f32 pattern 0x00400000; 0xff is NaN). Equal in bits to the rows
// kernel's product of the two decodes for every (code, scale) pair.
inline float mxfp8_quad_value(uint cw, uint sw, uint gg) {
    const uint c = (cw >> (8u * gg)) & 0xffu;
    const uint sb = (sw >> (8u * gg)) & 0xffu;
    const uint mag = c & 0x7fu;
    const float e = float(as_type<half>(ushort(mag << 7))) * 256.0f;
    const float m = mag == 0x7fu ? NAN : e;
    const float signed_m = as_type<float>(as_type<uint>(m) ^ ((c & 0x80u) << 24));
    const uint sbits = sb == 0xffu ? 0x7fc00000u : (sb == 0u ? 0x00400000u : (sb << 23));
    return signed_m * as_type<float>(sbits);
}

kernel void mxfp8_quads_gemv(device const uint8_t *W [[buffer(0)]],
                             device const float *x [[buffer(1)]],
                             device float *y [[buffer(2)]],
                             constant Nvfp4Args &args [[buffer(3)]],
                             device atomic_uint *faults [[buffer(4)]],
                             device const int *ids [[buffer(5)]],
                             device const float *scales [[buffer(6)]],
                             device const float *route_w [[buffer(7)]],
                             uint2 tg [[threadgroup_position_in_grid]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint sg [[simdgroup_index_in_threadgroup]])
{
    const uint r = tg.x * 8u + sg;
    const uint slot = tg.y;
    if (r >= args.rows || slot >= args.slots) {
        return;
    }
    const uint out = nv_out(args, slot, r);
    ulong expert = 0;
    float w_scale = 0.f;
    if (!nv_route(ids, scales, route_w, args, slot, expert, w_scale)) {
        if (lane == 0u) {
            nv_bad_slot(y, args, faults, out);
        }
        return;
    }
    x += slot * args.x_stride;
    const uint blocks = args.k / 128u;
    device const uint8_t *wr = W + (expert * (ulong)args.rows + (ulong)r) * (ulong)blocks * 132ul;
    bool bad = false;
    float acc = 0.f;
    for (uint b = 0u; b < blocks; ++b) {
        device const uint8_t *blk = wr + (ulong)b * 132ul;
        const uint cw = *(device const uint *)(blk + 4u * lane);
        const uint sw = *(device const uint *)(blk + 128u);
        for (uint gg = 0u; gg < 4u; ++gg) {
            const float v = mxfp8_quad_value(cw, sw, gg);
            // v is inf or NaN: its exponent field is all ones.
            bad |= (as_type<uint>(v) << 1) >= 0xff000000u;
            acc += v * x[(4u * b + gg) * 32u + lane];
        }
    }
    const float dot = simd_sum(acc) * w_scale;
    const bool usable = simd_all(!bad);
    if (!usable) {
        if (lane == 0u) {
            raise(faults, args.layer, BLOOMERY_FAULT_SITE_QUANT_COLUMN);
            y[out] = NAN;
        }
        return;
    }
    if (!isfinite(dot) && lane == 0u) {
        raise(faults, args.layer, BLOOMERY_FAULT_SITE_F32_PRODUCT);
    }
    if (lane == 0u) {
        y[out] = isfinite(dot) ? dot : NAN;
    }
}
