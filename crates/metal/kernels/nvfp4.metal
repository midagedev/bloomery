// The NVFP4 mat-vec of Inkling 2's routed experts: one SIMD group a row
// over the blocks of the fused stacks the family's files hold their first
// twenty stored layers' experts in, the per-expert global scale a launch
// word applied after the dot.
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
// The rows entry (nvfp4_rows_gemv) is bloomery's NVFP4 rows type, the
// MLX folder's bytes as stored: 9 bytes a 16-value group, the UE4M3 scale
// byte then 8 code bytes, value 2i the low nibble of code byte i and 2i + 1
// its high, the tensor's scale one launch word. A lane takes the groups
// lane, lane + 32, ... of a row, each group's eight bytes in order.

// The per-expert global scale (the file's `<stack>.scale` rows, one f32 an
// expert) multiplies each row's finished dot once — the same elementwise
// product either way, exact, so its placement costs nothing.
//
// No silent failure: a scale byte that is no UE4M3 (bit 7 set, or E4M3's
// NaN 0x7f) has no defined scale; the decode is NaN, the quant-column site
// raises and the row is NaN — never a plausible dot. A row whose dot stops
// being finite raises the product site.
//
// This file comes after fault.metal (raise).

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
    uint rows;     // the output's rows
    uint layer;
    float w_scale; // the expert's global scale, the launch's own word
};

static_assert(sizeof(Nvfp4Args) == 16, "Nvfp4Args is 16 bytes: the host packs four words");

kernel void nvfp4_gemv(device const block_nvfp4 *W [[buffer(0)]],
                       device const float *x [[buffer(1)]],
                       device float *y [[buffer(2)]],
                       constant Nvfp4Args &args [[buffer(3)]],
                       device atomic_uint *faults [[buffer(4)]],
                       uint row [[threadgroup_position_in_grid]],
                       uint lane [[thread_index_in_simdgroup]],
                       uint sg [[simdgroup_index_in_threadgroup]])
{
    const uint r = row * 8u + sg;
    if (r >= args.rows) {
        return;
    }
    const uint blocks = args.k / NV_VALUES;
    device const block_nvfp4 *wr = W + (ulong)r * (ulong)blocks;
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
    const float dot = simd_sum(acc) * args.w_scale;
    // One lane's scale byte that is none makes the whole row undefined.
    usable = simd_all(usable);
    if (!usable) {
        if (lane == 0u) {
            raise(faults, args.layer, BLOOMERY_FAULT_SITE_QUANT_COLUMN);
            y[r] = NAN;
        }
        return;
    }
    if (!isfinite(dot) && lane == 0u) {
        raise(faults, args.layer, BLOOMERY_FAULT_SITE_F32_PRODUCT);
    }
    if (lane == 0u) {
        y[r] = isfinite(dot) ? dot : NAN;
    }
}

// The rows entry: a SIMD group a row over `k / 16` nine-byte groups.
kernel void nvfp4_rows_gemv(device const uint8_t *W [[buffer(0)]],
                            device const float *x [[buffer(1)]],
                            device float *y [[buffer(2)]],
                            constant Nvfp4Args &args [[buffer(3)]],
                            device atomic_uint *faults [[buffer(4)]],
                            uint row [[threadgroup_position_in_grid]],
                            uint lane [[thread_index_in_simdgroup]],
                            uint sg [[simdgroup_index_in_threadgroup]])
{
    const uint r = row * 8u + sg;
    if (r >= args.rows) {
        return;
    }
    const uint groups = args.k / 16u;
    device const uint8_t *wr = W + (ulong)r * (ulong)groups * 9ul;
    bool usable = true;
    float acc = 0.f;
    for (uint g = lane; g < groups; g += 32u) {
        device const uint8_t *grp = wr + (ulong)g * 9ul;
        const float s = nvfp4_scale_half(uint(grp[0]));
        usable = usable && isfinite(s);
        for (uint i = 0u; i < 8u; ++i) {
            const uint byte = uint(grp[1u + i]);
            const uint col = g * 16u + 2u * i;
            acc += (NV_KVALUES[byte & 0x0fu] * s) * x[col];
            acc += (NV_KVALUES[byte >> 4u] * s) * x[col + 1u];
        }
    }
    const float dot = simd_sum(acc) * args.w_scale;
    usable = simd_all(usable);
    if (!usable) {
        if (lane == 0u) {
            raise(faults, args.layer, BLOOMERY_FAULT_SITE_QUANT_COLUMN);
            y[r] = NAN;
        }
        return;
    }
    if (!isfinite(dot) && lane == 0u) {
        raise(faults, args.layer, BLOOMERY_FAULT_SITE_F32_PRODUCT);
    }
    if (lane == 0u) {
        y[r] = isfinite(dot) ? dot : NAN;
    }
}

// The two other forms a routed stack of the family's files takes, beside the
// NVFP4 ones, each with the same launch word: the GGUF coda's Q8_0 blocks
// (q8_0_scaled_gemv) and the MLX folder coda's MXFP8 rows
// (mxfp8_rows_gemv). The launch word is the scale of the row's finished dot
// (a route weight, the coda's experts having no global scale), so a routed
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
                             uint row [[threadgroup_position_in_grid]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint sg [[simdgroup_index_in_threadgroup]])
{
    const uint r = row * 8u + sg;
    if (r >= args.rows) {
        return;
    }
    const uint blocks = args.k / 32u;
    device const uint8_t *wr = W + (ulong)r * (ulong)blocks * 34ul;
    bool usable = true;
    float acc = 0.f;
    for (uint i = 0u; i < blocks; ++i) {
        device const uint8_t *blk = wr + (ulong)i * 34ul;
        const float d = q8_0_scale(uint(blk[0]), uint(blk[1]));
        usable = usable && isfinite(d);
        const float v = float(as_type<char>(blk[2u + lane])) * d;
        acc += v * x[i * 32u + lane];
    }
    const float dot = simd_sum(acc) * args.w_scale;
    usable = simd_all(usable);
    if (!usable) {
        if (lane == 0u) {
            raise(faults, args.layer, BLOOMERY_FAULT_SITE_QUANT_COLUMN);
            y[r] = NAN;
        }
        return;
    }
    if (!isfinite(dot) && lane == 0u) {
        raise(faults, args.layer, BLOOMERY_FAULT_SITE_F32_PRODUCT);
    }
    if (lane == 0u) {
        y[r] = isfinite(dot) ? dot : NAN;
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
                            uint row [[threadgroup_position_in_grid]],
                            uint lane [[thread_index_in_simdgroup]],
                            uint sg [[simdgroup_index_in_threadgroup]])
{
    const uint r = row * 8u + sg;
    if (r >= args.rows) {
        return;
    }
    const uint groups = args.k / 32u;
    device const uint8_t *wr = W + (ulong)r * (ulong)groups * 33ul;
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
    const float dot = simd_sum(acc) * args.w_scale;
    usable = simd_all(usable);
    if (!usable) {
        if (lane == 0u) {
            raise(faults, args.layer, BLOOMERY_FAULT_SITE_QUANT_COLUMN);
            y[r] = NAN;
        }
        return;
    }
    if (!isfinite(dot) && lane == 0u) {
        raise(faults, args.layer, BLOOMERY_FAULT_SITE_F32_PRODUCT);
    }
    if (lane == 0u) {
        y[r] = isfinite(dot) ? dot : NAN;
    }
}
