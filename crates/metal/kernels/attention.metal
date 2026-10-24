// Decode attention for one grouped-query layer: 16 query heads, 2 key heads,
// head dim 256. A threadgroup is one (key head, segment) of eight SIMD
// groups; each SIMD group holds one query head, and its 32 lanes split the
// head, 8 dims a lane. The segment's keys stream from the cache straight
// into registers, one 16-byte read a key a lane; there is no threadgroup
// tile.
//
// The segment that owns the position appends that key and value. The whole
// owning SIMD group builds each row: the K row is normed and roped through
// threadgroup scratch, and every lane converts and stores its own 8 values,
// so the row lands in one pass instead of one lane's 256 serial stores. The
// q8_0 append shares the row out a 4-lane quad a 32-value block.
//
// One segment is the flash alone: the epilogue normalizes, applies the gate
// and writes the output. More than one writes a partial per segment, and
// the last threadgroup of the dispatch — the ticket's — folds those
// partials, applies the gate and writes the output, in the merge's
// arithmetic, then resets the ticket. There is no merge dispatch.
//
// Numeric order of the norm and the partial rope, from rope_neox's
// head-256 partial append. Square pairs are (i, i + 32) for i < 32 and
// (tid + 32, tid + 128) otherwise. The rope turns the first 64 values; the
// other 192 pass through. The CUDA sum is f64; this GPU has no f64, so the
// same pairing, the same xor butterfly and the same warp order are f32.
// The rope pair is that kernel's fma form: y0 = fma(n0, c, -(n1 * s)),
// y1 = fma(n0, s, n1 * c), table row pos * 64, pair i at 2i, 2i + 1.
// A q8_0 append is gguf quantize_q8_0: scale f16(amax / 127), codes
// nearest_int(v * id) with id = 127 / amax (0 when amax is 0 or id
// overflows).
//
// The score is each lane's 8 fmas in dim order, then the SIMD group's
// built-in sum (at most 31 adds, whichever tree the language picks), then
// the exact scale, as the band in the oracle derives. The softmax rescales
// once a 16-key tile; the fold adds the segments' partials in ascending
// segment order. The value sum is one fma a key in key order, per dim.
//
// The rows launches run a chunk of causal queries. attn_norm_rope_rows appends
// row r's key and value at position from + r, by the same append the decode
// flash runs, one SIMD group a (key head, row). gqa256_flash_f16_rows is the
// decode flash with a row grid dimension: row r is the launch at position
// from + r over from + r + 1 keys, with its own partials and ticket word, and
// it reads the rows the append dispatch wrote. A row is the bits of the decode
// launch at its position with the same segment count.
//
// The gate is the second half of each query head, the raw projection. The
// output value is attn * (1 / (1 + exp(-g))), after the softmax. A non-finite
// gate, or a non-finite product of a finite attention, raises the product
// site and stores NaN: a non-finite gate can otherwise leave the attention
// itself, which is a plausible value. A non-finite attention stores NaN and
// raises nothing. That fault is already the quant site, and the product site
// is the smaller code, so raising it would replace the quant site.

#pragma METAL fp contract(off)

static_assert(BLOOMERY_ATTN_GQA == BLOOMERY_ATTN_Q / BLOOMERY_ATTN_KV,
              "eight query heads share a key head");
static_assert(BLOOMERY_ATTN_HPT * BLOOMERY_ATTN_SG == BLOOMERY_ATTN_GQA,
              "eight groups of one head cover a key head");
static_assert(BLOOMERY_ATTN_PARTIAL == BLOOMERY_ATTN_HEAD + 2u,
              "a partial is the max, the weight sum and the head");
static_assert(BLOOMERY_ATTN_QG == BLOOMERY_ATTN_HEAD * 2u,
              "a query head is the query and then the gate");
static_assert(BLOOMERY_ATTN_THREADS == BLOOMERY_ATTN_LANES * BLOOMERY_ATTN_SG,
              "the flash threadgroup is eight SIMD groups");
static_assert(BLOOMERY_ATTN_LANES * 8u == BLOOMERY_ATTN_HEAD,
              "a lane's 8 values cover the head");
static_assert(BLOOMERY_ATTN_Q == BLOOMERY_ATTN_THREADS / 16u,
              "the folding group covers the heads at 16 dims a thread");

struct AttnArgs {
    uint segments;
    uint ctx;
    uint layer;
    float eps;
    float scale;
    uint pos;
    uint keys;
};

static_assert(sizeof(AttnArgs) == 28, "AttnArgs is 28 bytes: the host packs seven words");

// The rows launches' argument block: a chunk of `rows` queries whose first
// sits at position `from`, row r at position from + r with from + r + 1 live
// keys. The strides are in floats between a row's query projection, raw key,
// raw value and output.
struct AttnRowsArgs {
    uint segments;
    uint ctx;
    uint layer;
    float eps;
    float scale;
    uint from;
    uint rows;
    uint q_stride;
    uint k_stride;
    uint v_stride;
    uint out_stride;
};

static_assert(sizeof(AttnRowsArgs) == 44, "AttnRowsArgs is 44 bytes: the host packs eleven words");

// The decode launch's words for row `row` of a rows launch.
inline AttnArgs attn_row_args(constant AttnRowsArgs &args, uint row) {
    AttnArgs a;
    a.segments = args.segments;
    a.ctx = args.ctx;
    a.layer = args.layer;
    a.eps = args.eps;
    a.scale = args.scale;
    a.pos = args.from + row;
    a.keys = a.pos + 1u;
    return a;
}

// The hardware half conversion. A finite pattern is the host `half_to_f32`.
// The cast clears the sign of a negative NaN; the source sign is put back,
// which does not change a finite value. A NaN's payload is not the host's.
inline float attn_half(ushort bits) {
    float y = float(as_type<half>(bits));
    uint sign = (uint(bits) & 0x8000u) << 16;
    return as_type<float>(as_type<uint>(y) | sign);
}

// `1 / (1 + exp(-x))`, the host `route_core::sigmoid`.
inline float attn_sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

// f32 to f16 bits, round to nearest even, the host `f32_to_f16_bits`.
static ushort attn_f32_to_f16(float x) {
    uint b = as_type<uint>(x);
    ushort sign = ushort((b >> 16) & 0x8000u);
    uint a = b & 0x7fffffffu;
    if (a > 0x7f800000u) {
        return sign | 0x7e00u;
    }
    int exp = int(a >> 23) - 127;
    uint frac = a & 0x007fffffu;
    if (exp > 15) {
        return sign | 0x7c00u;
    }
    if (exp >= -14) {
        uint v = (uint(exp + 15) << 23) | frac;
        uint t = v + 0x0fffu + ((v >> 13) & 1u);
        uint h = t >> 13;
        if ((h & 0x7c00u) == 0x7c00u) {
            return sign | 0x7c00u;
        }
        return sign | ushort(h);
    }
    if (exp < -25) {
        return sign;
    }
    uint shift = uint(-1 - exp);
    uint v = 0x00800000u | frac;
    uint tie = 1u << (shift - 1u);
    uint rem = v & ((1u << shift) - 1u);
    uint h = v >> shift;
    if (rem > tie || (rem == tie && (h & 1u) == 1u)) {
        h += 1u;
    }
    return sign | ushort(h);
}

// ggml's nearest_int, the host `nearest_int`.
static int attn_nearest_int(float fval) {
    float val = fval + 12582912.0f;
    uint i = as_type<uint>(val);
    return int(i & 0x007fffffu) - 0x00400000;
}

static float attn_dequant(uint word, ushort scale, uint dim) {
    uint byte = (word >> (8u * (dim % 4u))) & 0xffu;
    int code = int(byte);
    if (code >= 128) {
        code -= 256;
    }
    return float(code) * attn_half(scale);
}

// Sum of squares in the partial-append pair tree, f32. Every lane returns
// the same sum. `scratch` holds the 256 values. The tree is the CUDA
// append's, over 128 threads of four warps; one SIMD group's 32 lanes
// evaluate it by walking the four warps' pairs themselves, so the warp
// count here is the tree's, not the launch's SIMD-group count.
static float attn_sum_squares(threadgroup float *scratch, uint lane) {
    constexpr uint pair_warps = 4u;
    float warp_acc[pair_warps];
    for (uint w = 0u; w < pair_warps; ++w) {
        uint tid = w * BLOOMERY_ATTN_LANES + lane;
        uint i0;
        uint i1;
        if (tid < BLOOMERY_ATTN_LANES) {
            i0 = tid;
            i1 = tid + BLOOMERY_ATTN_LANES;
        } else {
            i0 = tid + BLOOMERY_ATTN_LANES;
            i1 = tid + BLOOMERY_ATTN_HEAD / 2u;
        }
        float s0 = scratch[i0] * scratch[i0];
        float s1 = scratch[i1] * scratch[i1];
        warp_acc[w] = s0 + s1;
    }
    for (uint mask = 16u; mask >= 1u; mask >>= 1u) {
        for (uint w = 0u; w < pair_warps; ++w) {
            warp_acc[w] += simd_shuffle_xor(warp_acc[w], ushort(mask));
        }
    }
    float sum = warp_acc[0];
    sum += warp_acc[1];
    sum += warp_acc[2];
    sum += warp_acc[3];
    return sum;
}

// RMS scale of a head whose squares already sum to `sum`. Not finite and
// positive becomes NaN, and the values it scales are NaN.
static float attn_rms_scale(float sum, float eps) {
    float mean = sum / float(BLOOMERY_ATTN_HEAD);
    float scale = 1.0f / sqrt(mean + eps);
    if (!(scale > 0.0f) || !(scale < INFINITY)) {
        return NAN;
    }
    return scale;
}

// Norm then partial rope of the head in `scratch`, into `y` (this lane's 8
// dims). The gain and the rope row are 256 and 64. `lane` owns dims
// lane*8 .. +7.
static void attn_norm_rope(threadgroup float *scratch,
                           device float const *gain,
                           device float const *rope_row,
                           uint lane,
                           float eps,
                           thread float y[8]) {
    simdgroup_barrier(mem_flags::mem_threadgroup);
    float scale = attn_rms_scale(attn_sum_squares(scratch, lane), eps);
    uint dim0 = lane * 8u;
    float nloc[8];
    for (uint d = 0u; d < 8u; ++d) {
        uint i = dim0 + d;
        nloc[d] = (scale * gain[i]) * scratch[i];
    }
    for (uint d = 0u; d < 8u; ++d) {
        scratch[dim0 + d] = nloc[d];
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    uint pairs = BLOOMERY_ATTN_ROT / 2u;
    for (uint d = 0u; d < 8u; ++d) {
        uint i = dim0 + d;
        if (i < pairs) {
            float n0 = scratch[i];
            float n1 = scratch[i + pairs];
            float c = rope_row[2u * i];
            float s = rope_row[2u * i + 1u];
            y[d] = fma(n0, c, -(n1 * s));
        } else if (i < BLOOMERY_ATTN_ROT) {
            float n0 = scratch[i - pairs];
            float n1 = scratch[i];
            float c = rope_row[2u * (i - pairs)];
            float s = rope_row[2u * (i - pairs) + 1u];
            y[d] = fma(n0, s, n1 * c);
        } else {
            y[d] = scratch[i];
        }
    }
}

// The lane's 8 values of a row, converted and stored: the row's 256 values
// land in one pass over the group's lanes, each value the same
// `f32_to_f16_bits` the host quantizer runs.
static void attn_store_f16_lane(thread float const row[8],
                                uint lane,
                                device ushort *cache,
                                uint cache_row) {
    device ushort *dst = cache + cache_row * BLOOMERY_ATTN_HEAD + lane * 8u;
    for (uint d = 0u; d < 8u; ++d) {
        dst[d] = attn_f32_to_f16(row[d]);
    }
}

// One q8_0 row, the block's four lanes to a 32-value block: each lane holds
// the block's values [lane*8, lane*8 + 8), the quad reduces the amax and
// the finiteness, and each lane quantizes and stores its two code words.
// `quantize_q8_0`'s rule; the amax is exact in any order, so the quad's
// bits are the host quantizer's. A non-finite block is a NaN scale and zero
// codes, and raises the quant site.
static void attn_store_q8_quad(thread float const row[8],
                               uint lane,
                               device uint *codes,
                               device ushort *scales,
                               uint cache_row,
                               device atomic_uint *faults,
                               uint layer) {
    float amax = 0.0f;
    uint bad = 0u;
    for (uint d = 0u; d < 8u; ++d) {
        if (!isfinite(row[d])) {
            bad = 1u;
        } else {
            amax = max(amax, abs(row[d]));
        }
    }
    // The quad's four amaxes and four bad flags, every lane.
    for (ushort mask = 1u; mask <= 2u; mask <<= 1u) {
        amax = max(amax, simd_shuffle_xor(amax, mask));
        bad = max(bad, simd_shuffle_xor(bad, mask));
    }
    uint block = lane / 4u;
    if (bad != 0u) {
        if (lane % 4u == 0u) {
            scales[cache_row * 8u + block] = 0x7e00u;
        }
        for (uint w = 0u; w < 2u; ++w) {
            codes[cache_row * 64u + lane * 2u + w] = 0u;
        }
        if (lane % 4u == 0u) {
            raise(faults, layer, BLOOMERY_FAULT_SITE_KV_QUANT);
        }
        return;
    }
    // The host quantizer's multiplier: 127 / amax, else 0.
    float id = (amax != 0.0f) ? 127.0f / amax : 0.0f;
    if (!isfinite(id)) {
        id = 0.0f;
    }
    if (lane % 4u == 0u) {
        scales[cache_row * 8u + block] = attn_f32_to_f16(amax / 127.0f);
    }
    for (uint w = 0u; w < 2u; ++w) {
        uint word = 0u;
        for (uint byte = 0u; byte < 4u; ++byte) {
            float v = row[w * 4u + byte];
            int code = clamp(attn_nearest_int(v * id), -128, 127);
            word |= (uint(code) & 0xffu) << (8u * byte);
        }
        codes[cache_row * 64u + lane * 2u + w] = word;
    }
}

static uint attn_span(uint live, uint segments) {
    if (live == 0u || segments == 0u) {
        return BLOOMERY_ATTN_TILE;
    }
    uint per = (live + segments - 1u) / segments;
    per = ((per + BLOOMERY_ATTN_TILE - 1u) / BLOOMERY_ATTN_TILE) * BLOOMERY_ATTN_TILE;
    return per < BLOOMERY_ATTN_TILE ? BLOOMERY_ATTN_TILE : per;
}

// Store `proposed` unless the gate or the product of a finite attention is
// not finite. Then the product site is raised and the stored value is NaN.
inline void attn_commit(device float *out,
                        uint index,
                        float attn,
                        float g,
                        float proposed,
                        device atomic_uint *faults,
                        uint layer) {
    float y = proposed;
    if (isfinite(attn)) {
        if (!isfinite(g) || !isfinite(y)) {
            raise(faults, layer, BLOOMERY_FAULT_SITE_F32_PRODUCT);
            y = NAN;
        }
    } else {
        y = NAN;
    }
    out[index] = y;
}

// The f16 cache. One tile of rows sits in threadgroup memory as 16-byte
// words: the group loads it once, cooperatively, and the eight SIMD groups
// read it a head at a time. The lane's 8 values of a tile row are one
// word, widened by the hardware conversion: a finite half is exact either
// way, and a NaN's sign does not change the attention's NaN-ness. A row
// past the live keys loads as zero bits, the value 0.0f.
struct F16Cache {
    device ushort *k;
    device ushort *v;
    threadgroup uint4 *tile_k;
    threadgroup uint4 *tile_v;

    void load(uint tid, uint kv, uint ctx, uint lo, uint hi) {
        constexpr uint words = BLOOMERY_ATTN_TILE * BLOOMERY_ATTN_HEAD / 8u;
        for (uint e = tid; e < words; e += BLOOMERY_ATTN_THREADS) {
            uint t = e / 32u;
            uint c = e - t * 32u;
            uint index = lo + t;
            uint4 wk = 0u;
            uint4 wv = 0u;
            if (index < hi) {
                uint row = kv * ctx + index;
                const device uint4 *pk = (const device uint4 *)(k + row * BLOOMERY_ATTN_HEAD);
                const device uint4 *pv = (const device uint4 *)(v + row * BLOOMERY_ATTN_HEAD);
                wk = pk[c];
                wv = pv[c];
            }
            tile_k[e] = wk;
            tile_v[e] = wv;
        }
    }

    void key(uint t, uint lane, thread float out[8]) {
        uint4 w = tile_k[t * 32u + lane];
        float4 f0 = float4(as_type<half4>(w.xy));
        float4 f1 = float4(as_type<half4>(w.zw));
        out[0] = f0.x;
        out[1] = f0.y;
        out[2] = f0.z;
        out[3] = f0.w;
        out[4] = f1.x;
        out[5] = f1.y;
        out[6] = f1.z;
        out[7] = f1.w;
    }

    void value(uint t, uint lane, thread float out[8]) {
        uint4 w = tile_v[t * 32u + lane];
        float4 f0 = float4(as_type<half4>(w.xy));
        float4 f1 = float4(as_type<half4>(w.zw));
        out[0] = f0.x;
        out[1] = f0.y;
        out[2] = f0.z;
        out[3] = f0.w;
        out[4] = f1.x;
        out[5] = f1.y;
        out[6] = f1.z;
        out[7] = f1.w;
    }

    void store_k(thread float const row[8], uint lane, uint cache_row, device atomic_uint *, uint) {
        attn_store_f16_lane(row, lane, k, cache_row);
    }

    void store_v(thread float const row[8], uint lane, uint cache_row, device atomic_uint *, uint) {
        attn_store_f16_lane(row, lane, v, cache_row);
    }
};

// The q8_0 cache, one tile in threadgroup memory: 64 code words and 8
// scales a row, low byte first. A row past the live keys loads as zero
// codes and a zero scale, the value 0.0f. The lane's 8 values sit in two
// code words of one 32-value block.
struct Q8Cache {
    device uint *k_codes;
    device ushort *k_scales;
    device uint *v_codes;
    device ushort *v_scales;
    threadgroup uint *tile_k_codes;
    threadgroup ushort *tile_k_scales;
    threadgroup uint *tile_v_codes;
    threadgroup ushort *tile_v_scales;

    void load(uint tid, uint kv, uint ctx, uint lo, uint hi) {
        for (uint e = tid; e < BLOOMERY_ATTN_TILE * 64u; e += BLOOMERY_ATTN_THREADS) {
            uint t = e / 64u;
            uint c = e - t * 64u;
            uint index = lo + t;
            uint w = 0u;
            if (index < hi) {
                w = k_codes[(kv * ctx + index) * 64u + c];
            }
            tile_k_codes[e] = w;
        }
        for (uint e = tid; e < BLOOMERY_ATTN_TILE * 64u; e += BLOOMERY_ATTN_THREADS) {
            uint t = e / 64u;
            uint c = e - t * 64u;
            uint index = lo + t;
            uint w = 0u;
            if (index < hi) {
                w = v_codes[(kv * ctx + index) * 64u + c];
            }
            tile_v_codes[e] = w;
        }
        if (tid < BLOOMERY_ATTN_TILE * 8u) {
            uint t = tid / 8u;
            uint b = tid - t * 8u;
            uint index = lo + t;
            ushort ks = 0u;
            ushort vs = 0u;
            if (index < hi) {
                uint row = kv * ctx + index;
                ks = k_scales[row * 8u + b];
                vs = v_scales[row * 8u + b];
            }
            tile_k_scales[tid] = ks;
            tile_v_scales[tid] = vs;
        }
    }

    void key(uint t, uint lane, thread float out[8]) {
        ushort s = tile_k_scales[t * 8u + lane / 4u];
        uint w0 = tile_k_codes[t * 64u + lane * 2u];
        uint w1 = tile_k_codes[t * 64u + lane * 2u + 1u];
        for (uint d = 0u; d < 4u; ++d) {
            out[d] = attn_dequant(w0, s, d);
        }
        for (uint d = 0u; d < 4u; ++d) {
            out[4u + d] = attn_dequant(w1, s, d);
        }
    }

    void value(uint t, uint lane, thread float out[8]) {
        ushort s = tile_v_scales[t * 8u + lane / 4u];
        uint w0 = tile_v_codes[t * 64u + lane * 2u];
        uint w1 = tile_v_codes[t * 64u + lane * 2u + 1u];
        for (uint d = 0u; d < 4u; ++d) {
            out[d] = attn_dequant(w0, s, d);
        }
        for (uint d = 0u; d < 4u; ++d) {
            out[4u + d] = attn_dequant(w1, s, d);
        }
    }

    void store_k(thread float const row[8], uint lane, uint cache_row, device atomic_uint *faults,
                 uint layer) {
        attn_store_q8_quad(row, lane, k_codes, k_scales, cache_row, faults, layer);
    }

    void store_v(thread float const row[8], uint lane, uint cache_row, device atomic_uint *faults,
                 uint layer) {
        attn_store_q8_quad(row, lane, v_codes, v_scales, cache_row, faults, layer);
    }
};

// The segment that owns the position appends that key and value. The whole
// owning SIMD group builds each row and stores it a lane at a time.
template <typename Cache>
static void attn_append(Cache cache,
                        thread const AttnArgs &args,
                        device float const *k,
                        device float const *v,
                        device float const *k_gain,
                        device float const *rope,
                        device atomic_uint *faults,
                        bool ok,
                        uint kv,
                        uint seg,
                        uint span,
                        uint sg,
                        uint lane,
                        threadgroup float *scratch) {
    if (sg != 0u) {
        return;
    }
    bool owns = ok && args.pos < args.keys && args.pos / span == seg;
    if (!owns) {
        return;
    }
    uint dim0 = lane * 8u;
    threadgroup float *row0 = scratch;
    device float const *rope_row = rope + args.pos * BLOOMERY_ATTN_ROT;
    for (uint d = 0u; d < 8u; ++d) {
        row0[dim0 + d] = k[kv * BLOOMERY_ATTN_HEAD + dim0 + d];
    }
    float y[8];
    attn_norm_rope(row0, k_gain, rope_row, lane, args.eps, y);
    cache.store_k(y, lane, kv * args.ctx + args.pos, faults, args.layer);
    for (uint d = 0u; d < 8u; ++d) {
        y[d] = v[kv * BLOOMERY_ATTN_HEAD + dim0 + d];
    }
    cache.store_v(y, lane, kv * args.ctx + args.pos, faults, args.layer);
}

// This group's query head, normed and partially roped, in `qv`.
static void attn_prepare_q(thread const AttnArgs &args,
                           device float const *q,
                           device float const *q_gain,
                           device float const *rope,
                           bool ok,
                           uint kv,
                           uint sg,
                           uint lane,
                           threadgroup float *scratch,
                           thread float qv[8]) {
    for (uint d = 0u; d < 8u; ++d) {
        qv[d] = 0.0f;
    }
    if (!ok) {
        return;
    }
    uint head = kv * BLOOMERY_ATTN_GQA + sg * BLOOMERY_ATTN_HPT;
    uint dim0 = lane * 8u;
    threadgroup float *row = scratch + sg * BLOOMERY_ATTN_HEAD;
    for (uint d = 0u; d < 8u; ++d) {
        row[dim0 + d] = q[head * BLOOMERY_ATTN_QG + dim0 + d];
    }
    attn_norm_rope(row, q_gain, rope + args.pos * BLOOMERY_ATTN_ROT, lane, args.eps, qv);
}

// Online softmax over this segment's tiles, the keys streamed a row a
// lane. The running max, weight sum and values stay in registers. No
// barrier: one SIMD group works one head over its own registers.
template <typename Cache>
static void attn_tile_loop(Cache cache,
                           thread const AttnArgs &args,
                           bool ok,
                           uint kv,
                           uint seg,
                           uint span,
                           uint max_live,
                           uint tid,
                           uint lane,
                           thread float qv[8],
                           thread float *running_m,
                           thread float *running_l,
                           thread float acc[8]) {
    *running_m = -INFINITY;
    *running_l = 0.0f;
    for (uint d = 0u; d < 8u; ++d) {
        acc[d] = 0.0f;
    }
    if (!ok) {
        return;
    }
    uint lo0 = seg * span;
    uint seg_hi = min((seg + 1u) * span, max_live);
    for (uint lo = lo0; lo < seg_hi; lo += BLOOMERY_ATTN_TILE) {
        uint hi = min(lo + BLOOMERY_ATTN_TILE, seg_hi);
        cache.load(tid, kv, args.ctx, lo, hi);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float scores[BLOOMERY_ATTN_TILE];
        bool live[BLOOMERY_ATTN_TILE];
        float tile_max = -INFINITY;
        uint nvalid = 0u;
        float kvv[8];
        for (uint t = 0u; t < BLOOMERY_ATTN_TILE; ++t) {
            uint index = lo + t;
            bool on = index < hi && index < args.keys;
            live[t] = on;
            float dot = 0.0f;
            if (on) {
                cache.key(t, lane, kvv);
                for (uint d = 0u; d < 8u; ++d) {
                    dot = fma(qv[d], kvv[d], dot);
                }
            }
            // `on` is the same for every lane. The sum still runs on the
            // lanes that loaded nothing, whose dot is zero.
            dot = simd_sum(dot);
            float score = dot * args.scale;
            scores[t] = 0.0f;
            if (on) {
                scores[t] = score;
                tile_max = max(tile_max, score);
                nvalid += 1u;
            }
        }
        if (nvalid == 0u) {
            continue;
        }
        float new_m = max(*running_m, tile_max);
        float old_m = *running_m;
        float factor = old_m > -INFINITY ? exp(old_m - new_m) : 0.0f;
        *running_l *= factor;
        for (uint d = 0u; d < 8u; ++d) {
            acc[d] *= factor;
        }
        *running_m = new_m;
        for (uint t = 0u; t < BLOOMERY_ATTN_TILE; ++t) {
            if (!live[t]) {
                continue;
            }
            cache.value(t, lane, kvv);
            float weight = exp(scores[t] - new_m);
            *running_l += weight;
            for (uint d = 0u; d < 8u; ++d) {
                acc[d] = fma(weight, kvv[d], acc[d]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// One segment writes the gated output. More than one writes the partial.
static void attn_epilogue(thread const AttnArgs &args,
                          device float const *q,
                          device coherent(device) float *partials,
                          device float *out,
                          device atomic_uint *faults,
                          bool ok,
                          uint kv,
                          uint seg,
                          uint sg,
                          uint lane,
                          thread float *running_m,
                          thread float *running_l,
                          thread float acc[8]) {
    uint dim0 = lane * 8u;
    uint head = kv * BLOOMERY_ATTN_GQA + sg * BLOOMERY_ATTN_HPT;
    if (args.segments == 1u) {
        uint o = head * BLOOMERY_ATTN_HEAD + dim0;
        if (!ok || !(*running_l > 0.0f) || !isfinite(*running_m)) {
            for (uint d = 0u; d < 8u; ++d) {
                out[o + d] = NAN;
            }
            return;
        }
        float inv = 1.0f / *running_l;
        for (uint d = 0u; d < 8u; ++d) {
            float attn = acc[d] * inv;
            float g = q[head * BLOOMERY_ATTN_QG + BLOOMERY_ATTN_HEAD + dim0 + d];
            float gated = attn * attn_sigmoid(g);
            attn_commit(out, o + d, attn, g, gated, faults, args.layer);
        }
        return;
    }
    uint base = (head * args.segments + seg) * BLOOMERY_ATTN_PARTIAL;
    if (!ok) {
        if (lane == 0u) {
            partials[base] = NAN;
            partials[base + 1u] = NAN;
        }
        for (uint d = 0u; d < 8u; ++d) {
            partials[base + 2u + dim0 + d] = NAN;
        }
        return;
    }
    if (lane == 0u) {
        partials[base] = *running_m;
        partials[base + 1u] = *running_l;
    }
    for (uint d = 0u; d < 8u; ++d) {
        partials[base + 2u + dim0 + d] = acc[d];
    }
}

// The merge, folded into the flash: the dispatch's last threadgroup folds
// every head's segment partials in ascending segment order, applies the
// gate and writes the output, then resets the ticket. The arithmetic is the
// merge dispatch's, per head, in two passes: the first walks the maxima and
// the weight sums and names each segment's rescale factor, the second
// replays the merge's fma order with those factors over the values. The
// running max at a segment is the prefix maximum of the earlier segments'
// maxima, so no factor's exp reads another's: the first pass's exps and
// both passes' loads are independent of the fma chains that consume them.
// The factors sit in the tile space the loop left, one pair a segment a
// head. A NaN partial (a refused position) writes NaN and stops. An empty
// partial is skipped.
//
// The reads are the ticket's coherent ones: the writers fence before they
// arrive, and the acquire fence plus a coherent read is the visibility the
// protocol gives (a plain read after the fence is not, on this hardware:
// it served the zero page).
static void attn_fold(device coherent(device) float *partials,
                      device atomic_uint *ticket,
                      thread const AttnArgs &args,
                      device float const *q,
                      device float *out,
                      device atomic_uint *faults,
                      uint tid,
                      threadgroup float *factors) {
    ticket_acquire();
    uint head = tid / 16u;
    uint sub = tid % 16u;
    uint dim0 = sub * 16u;
    threadgroup float *mine = factors + head * args.segments * 2u;
    float m = -INFINITY;
    float l = 0.0f;
    bool refused = false;
    for (uint seg = 0u; seg < args.segments; ++seg) {
        uint base = (head * args.segments + seg) * BLOOMERY_ATTN_PARTIAL;
        float mj = ticket_load(partials, base);
        float lj = ticket_load(partials, base + 1u);
        if (isnan(mj) || isnan(lj)) {
            refused = true;
            break;
        }
        if (!(lj > 0.0f)) {
            continue;
        }
        if (mj > m) {
            float f = m > -INFINITY ? exp(m - mj) : 0.0f;
            if (sub == 0u) {
                // Up: the values enter raw and the accumulator rescales. The
                // kind word is one, so a first segment's factor of zero is a
                // factor, not the other branch.
                mine[2u * seg] = 1.0f;
                mine[2u * seg + 1u] = f;
            }
            m = mj;
            l = fma(l, f, lj);
        } else {
            float g = exp(mj - m);
            if (sub == 0u) {
                // Down: the values rescale and the accumulator stays.
                mine[2u * seg] = 0.0f;
                mine[2u * seg + 1u] = g;
            }
            l = fma(lj, g, l);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float acc[16];
    for (uint d = 0u; d < 16u; ++d) {
        acc[d] = 0.0f;
    }
    if (!refused) {
        for (uint seg = 0u; seg < args.segments; ++seg) {
            uint base = (head * args.segments + seg) * BLOOMERY_ATTN_PARTIAL;
            float lj = ticket_load(partials, base + 1u);
            if (isnan(lj) || !(lj > 0.0f)) {
                continue;
            }
            bool up = mine[2u * seg] != 0.0f;
            float f = mine[2u * seg + 1u];
            for (uint d = 0u; d < 16u; ++d) {
                float vj = ticket_load(partials, base + 2u + dim0 + d);
                if (up) {
                    acc[d] = fma(acc[d], f, vj);
                } else {
                    acc[d] = fma(vj, f, acc[d]);
                }
            }
        }
    }
    uint o = head * BLOOMERY_ATTN_HEAD + dim0;
    if (refused || !(l > 0.0f) || !isfinite(m)) {
        for (uint d = 0u; d < 16u; ++d) {
            out[o + d] = NAN;
        }
    } else {
        float inv = 1.0f / l;
        for (uint d = 0u; d < 16u; ++d) {
            float attn = acc[d] * inv;
            float gate = q[head * BLOOMERY_ATTN_QG + BLOOMERY_ATTN_HEAD + dim0 + d];
            float merged = attn * attn_sigmoid(gate);
            attn_commit(out, o + d, attn, gate, merged, faults, args.layer);
        }
    }
    if (tid == 0u) {
        ticket_reset(ticket);
    }
}

// Whether the position and the key count are inside the cache, and the live
// key count. A key count of 0 or past the cache raises the key-count site, a
// position at or past it the cache-position site, once per dispatch's
// threadgroup.
static bool attn_bounds(thread const AttnArgs &args,
                        device atomic_uint *faults,
                        bool active,
                        uint tid,
                        thread uint &max_live) {
    bool key_bad = args.keys == 0u || args.keys > args.ctx;
    bool pos_bad = args.pos >= args.ctx;
    max_live = key_bad ? 0u : args.keys;
    if (tid == 0u && active) {
        if (key_bad) {
            raise(faults, args.layer, BLOOMERY_FAULT_SITE_KEY_COUNT);
        }
        if (pos_bad) {
            raise(faults, args.layer, BLOOMERY_FAULT_SITE_CACHE_POS);
        }
    }
    return active && !key_bad && !pos_bad;
}

// The flash of one position. APPEND is the decode's: the owning segment
// appends this position's key and value first. The rows launch has appended
// them already (attn_norm_rope_rows) and passes no key, value or key gain.
template <typename Cache, bool APPEND>
static void attn_flash(Cache cache,
                       thread const AttnArgs &args,
                       device float const *q,
                       device float const *k,
                       device float const *v,
                       device float const *q_gain,
                       device float const *k_gain,
                       device float const *rope,
                       device coherent(device) float *partials,
                       device float *out,
                       device atomic_uint *faults,
                       device atomic_uint *ticket,
                       uint groups,
                       uint group,
                       uint tid,
                       threadgroup float *scratch,
                       threadgroup uint &last_flag) {
    uint sg = tid / BLOOMERY_ATTN_LANES;
    uint lane = tid % BLOOMERY_ATTN_LANES;
    uint kv = group / args.segments;
    uint seg = group - kv * args.segments;
    bool active = kv < BLOOMERY_ATTN_KV && seg < args.segments;

    uint max_live;
    bool ok = attn_bounds(args, faults, active, tid, max_live);

    uint span = attn_span(max_live, args.segments);
    if constexpr (APPEND) {
        if (active) {
            attn_append(cache, args, k, v, k_gain, rope, faults, ok, kv, seg, span, sg, lane,
                        scratch);
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    }

    if (active) {
        float qv[8];
        attn_prepare_q(args, q, q_gain, rope, ok, kv, sg, lane, scratch, qv);
        float running_m;
        float running_l;
        float acc[8];
        // The tiles load over the scratch the prepare paired through, so
        // every thread's scratch reads end here.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        attn_tile_loop(cache, args, ok, kv, seg, span, max_live, tid, lane, qv, &running_m,
                       &running_l, acc);
        attn_epilogue(args, q, partials, out, faults, ok, kv, seg, sg, lane, &running_m,
                      &running_l, acc);
    }

    // Past one segment the last threadgroup folds the partials; the ticket
    // is the dispatch's own, and the fold resets it for the next launch.
    if (args.segments > 1u) {
        ticket_is_last(ticket, groups, tid, last_flag);
        if (last_flag == 1u) {
            attn_fold(partials, ticket, args, q, out, faults, tid, (threadgroup float *)scratch);
        }
    }
}

// The threadgroup's one shared block, in 16-byte words: the scratch the
// append and the q prepare pair through, then — after the barrier that ends
// the prepare — the K and V tiles the loop loads over it. The scratch is
// the first half as floats; tile_v is the second half, so the two uses
// never share a word.
kernel void gqa256_flash_f16(device float const *q [[buffer(0)]],
                             device float const *k [[buffer(1)]],
                             device float const *v [[buffer(2)]],
                             device float const *q_gain [[buffer(3)]],
                             device float const *k_gain [[buffer(4)]],
                             device float const *rope [[buffer(5)]],
                             device ushort *cache_k [[buffer(6)]],
                             device ushort *cache_v [[buffer(7)]],
                             device coherent(device) float *partials [[buffer(8)]],
                             device float *out [[buffer(9)]],
                             constant AttnArgs &launch [[buffer(10)]],
                             device atomic_uint *faults [[buffer(11)]],
                             device atomic_uint *ticket [[buffer(12)]],
                             uint tid [[thread_index_in_threadgroup]],
                             uint3 group [[threadgroup_position_in_grid]],
                             uint3 tpg [[threadgroups_per_grid]]) {
    threadgroup uint4 shared[BLOOMERY_ATTN_TILE * BLOOMERY_ATTN_HEAD / 4];
    threadgroup uint last_flag = 0u;
    AttnArgs args = launch;
    F16Cache cache;
    cache.k = cache_k;
    cache.v = cache_v;
    cache.tile_k = shared;
    cache.tile_v = shared + BLOOMERY_ATTN_TILE * BLOOMERY_ATTN_HEAD / 8;
    attn_flash<F16Cache, true>(cache, args, q, k, v, q_gain, k_gain, rope, partials, out, faults,
                               ticket, tpg.x, group.x, tid, (threadgroup float *)shared,
                               last_flag);
}

kernel void gqa256_flash_q8(device float const *q [[buffer(0)]],
                            device float const *k [[buffer(1)]],
                            device float const *v [[buffer(2)]],
                            device float const *q_gain [[buffer(3)]],
                            device float const *k_gain [[buffer(4)]],
                            device float const *rope [[buffer(5)]],
                            device uint *k_codes [[buffer(6)]],
                            device ushort *k_scales [[buffer(7)]],
                            device uint *v_codes [[buffer(8)]],
                            device ushort *v_scales [[buffer(9)]],
                            device coherent(device) float *partials [[buffer(10)]],
                            device float *out [[buffer(11)]],
                            constant AttnArgs &launch [[buffer(12)]],
                            device atomic_uint *faults [[buffer(13)]],
                            device atomic_uint *ticket [[buffer(14)]],
                            uint tid [[thread_index_in_threadgroup]],
                            uint3 group [[threadgroup_position_in_grid]],
                            uint3 tpg [[threadgroups_per_grid]]) {
    // The q8_0 tiles are 544 bytes a row pair; the shared block covers the
    // f16 tiles' 16 KB, which the scratch also pairs through.
    threadgroup uint4 shared[BLOOMERY_ATTN_TILE * BLOOMERY_ATTN_HEAD / 4];
    threadgroup uint last_flag = 0u;
    AttnArgs args = launch;
    Q8Cache cache;
    cache.k_codes = k_codes;
    cache.k_scales = k_scales;
    cache.v_codes = v_codes;
    cache.v_scales = v_scales;
    cache.tile_k_codes = (threadgroup uint *)shared;
    cache.tile_k_scales = (threadgroup ushort *)(shared + BLOOMERY_ATTN_TILE * 16);
    cache.tile_v_codes = (threadgroup uint *)(shared + BLOOMERY_ATTN_TILE * 20);
    cache.tile_v_scales = (threadgroup ushort *)(shared + BLOOMERY_ATTN_TILE * 36);
    attn_flash<Q8Cache, true>(cache, args, q, k, v, q_gain, k_gain, rope, partials, out, faults,
                              ticket, tpg.x, group.x, tid, (threadgroup float *)shared,
                              last_flag);
}

// The append of a chunk of rows to the f16 cache: row r's key is normed and
// roped at position from + r and stored with its value, by the decode's own
// append, one SIMD group a (key head, row). The flash over the chunk
// (gqa256_flash_f16_rows) reads these rows, so it runs in a later dispatch.
kernel void attn_norm_rope_rows(device float const *k [[buffer(0)]],
                                device float const *v [[buffer(1)]],
                                device float const *k_gain [[buffer(2)]],
                                device float const *rope [[buffer(3)]],
                                device ushort *cache_k [[buffer(4)]],
                                device ushort *cache_v [[buffer(5)]],
                                constant AttnRowsArgs &args [[buffer(6)]],
                                device atomic_uint *faults [[buffer(7)]],
                                uint lane [[thread_index_in_threadgroup]],
                                uint3 group [[threadgroup_position_in_grid]]) {
    threadgroup float scratch[BLOOMERY_ATTN_HEAD];
    F16Cache cache;
    cache.k = cache_k;
    cache.v = cache_v;
    cache.tile_k = nullptr;
    cache.tile_v = nullptr;
    uint kv = group.x;
    uint row = group.y;
    AttnArgs a = attn_row_args(args, row);
    uint max_live;
    bool ok = attn_bounds(a, faults, true, lane, max_live);
    // One segment owns every position: its span covers the live keys.
    uint span = attn_span(max_live, 1u);
    attn_append(cache, a, k + row * args.k_stride, v + row * args.v_stride, k_gain, rope, faults,
                ok, kv, 0u, span, 0u, lane, scratch);
}

// The flash of a chunk of rows over the f16 cache: the decode flash with a
// row grid dimension, row r at position from + r with from + r + 1 live keys,
// each row the bits of the decode launch at its position. The grid is
// (key heads x segments, rows); every row has its own partials and ticket
// word.
kernel void gqa256_flash_f16_rows(device float const *q [[buffer(0)]],
                                  device float const *q_gain [[buffer(1)]],
                                  device float const *rope [[buffer(2)]],
                                  device ushort *cache_k [[buffer(3)]],
                                  device ushort *cache_v [[buffer(4)]],
                                  device coherent(device) float *partials [[buffer(5)]],
                                  device float *out [[buffer(6)]],
                                  constant AttnRowsArgs &args [[buffer(7)]],
                                  device atomic_uint *faults [[buffer(8)]],
                                  device atomic_uint *ticket [[buffer(9)]],
                                  uint tid [[thread_index_in_threadgroup]],
                                  uint3 group [[threadgroup_position_in_grid]],
                                  uint3 tpg [[threadgroups_per_grid]]) {
    threadgroup uint4 shared[BLOOMERY_ATTN_TILE * BLOOMERY_ATTN_HEAD / 4];
    threadgroup uint last_flag = 0u;
    F16Cache cache;
    cache.k = cache_k;
    cache.v = cache_v;
    cache.tile_k = shared;
    cache.tile_v = shared + BLOOMERY_ATTN_TILE * BLOOMERY_ATTN_HEAD / 8;
    uint row = group.y;
    AttnArgs a = attn_row_args(args, row);
    uint partials_stride = BLOOMERY_ATTN_Q * args.segments * BLOOMERY_ATTN_PARTIAL;
    attn_flash<F16Cache, false>(cache, a, q + row * args.q_stride, nullptr, nullptr, q_gain,
                                nullptr, rope, partials + row * partials_stride,
                                out + row * args.out_stride, faults, ticket + row, tpg.x, group.x,
                                tid, (threadgroup float *)shared, last_flag);
}
