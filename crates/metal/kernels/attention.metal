// Decode attention for one grouped-query layer: 16 query heads, 2 key heads,
// head dim 256. A threadgroup is one (key head, segment). It holds one tile
// of 16 K/V rows in threadgroup memory and four SIMD groups (two query heads
// each) read that tile.
//
// One segment normalizes, applies the gate and writes the layer output. More
// than one writes a partial per segment. The merge folds those partials in
// segment order, applies the gate and writes the output. The caller places
// the barrier between the two dispatches.
//
// Numeric order of the norm and the partial rope, from rope_neox's
// head-256 partial append. Square pairs are (i, i + 32) for i < 32 and
// (tid + 32, tid + 128) otherwise. The rope turns the first 64 values; the
// other 192 pass through. The CUDA sum is f64; this GPU has no f64, so the
// same pairing, the same xor butterfly and the same warp order are f32.
// The rope pair is that kernel's fma form: y0 = fma(n0, c, -(n1 * s)),
// y1 = fma(n0, s, n1 * c), table row pos * 64, pair i at 2i, 2i + 1.
// A q8_0 append is gguf quantize_q8_0: scale f16(amax / 127), codes
// nearest_int(v * id) with id = 127 / amax (0 when amax is 0 or id overflows).
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
              "four groups of two heads cover a key head");
static_assert(BLOOMERY_ATTN_PARTIAL == BLOOMERY_ATTN_HEAD + 2u,
              "a partial is the max, the weight sum and the head");
static_assert(BLOOMERY_ATTN_QG == BLOOMERY_ATTN_HEAD * 2u,
              "a query head is the query and then the gate");
static_assert(BLOOMERY_ATTN_THREADS == BLOOMERY_ATTN_LANES * BLOOMERY_ATTN_SG,
              "the flash threadgroup is four SIMD groups");
static_assert(BLOOMERY_ATTN_LANES * 8u == BLOOMERY_ATTN_HEAD,
              "a lane's 8 values cover the head");

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
// the same sum. `scratch` holds the 256 values.
static float attn_sum_squares(threadgroup float *scratch, uint lane) {
    float warp_acc[BLOOMERY_ATTN_SG];
    for (uint w = 0u; w < BLOOMERY_ATTN_SG; ++w) {
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
        for (uint w = 0u; w < BLOOMERY_ATTN_SG; ++w) {
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

static void attn_store_f16(threadgroup float *row, device ushort *cache, uint cache_row) {
    for (uint i = 0u; i < BLOOMERY_ATTN_HEAD; ++i) {
        cache[cache_row * BLOOMERY_ATTN_HEAD + i] = attn_f32_to_f16(row[i]);
    }
}

// One q8_0 row: eight blocks, `quantize_q8_0`'s rule. A non-finite block is
// a NaN scale and zero codes, and raises the quant site.
static void attn_store_q8(threadgroup float *row,
                          device uint *codes,
                          device ushort *scales,
                          uint cache_row,
                          device atomic_uint *faults,
                          uint layer) {
    for (uint b = 0u; b < 8u; ++b) {
        bool bad = false;
        float amax = 0.0f;
        for (uint i = 0u; i < 32u; ++i) {
            float v = row[b * 32u + i];
            if (!isfinite(v)) {
                bad = true;
            } else {
                amax = max(amax, abs(v));
            }
        }
        uint word0 = cache_row * 64u + b * 8u;
        if (bad) {
            scales[cache_row * 8u + b] = 0x7e00u;
            for (uint w = 0u; w < 8u; ++w) {
                codes[word0 + w] = 0u;
            }
            raise(faults, layer, BLOOMERY_FAULT_SITE_KV_QUANT);
            continue;
        }
        // The host quantizer's multiplier: 127 / amax, else 0.
        float id = (amax != 0.0f) ? 127.0f / amax : 0.0f;
        if (!isfinite(id)) {
            id = 0.0f;
        }
        scales[cache_row * 8u + b] = attn_f32_to_f16(amax / 127.0f);
        for (uint w = 0u; w < 8u; ++w) {
            uint word = 0u;
            for (uint byte = 0u; byte < 4u; ++byte) {
                float v = row[b * 32u + w * 4u + byte];
                int code = clamp(attn_nearest_int(v * id), -128, 127);
                word |= (uint(code) & 0xffu) << (8u * byte);
            }
            codes[word0 + w] = word;
        }
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

// The f16 cache. The tile is the threadgroup copy of one K/V tile.
struct F16Cache {
    device ushort *k;
    device ushort *v;
    threadgroup ushort *tile_k;
    threadgroup ushort *tile_v;

    void load(uint tid, uint kv, uint ctx, uint lo, uint hi) {
        for (uint e = tid; e < BLOOMERY_ATTN_TILE * BLOOMERY_ATTN_HEAD; e += BLOOMERY_ATTN_THREADS) {
            uint t = e / BLOOMERY_ATTN_HEAD;
            uint dim = e - t * BLOOMERY_ATTN_HEAD;
            uint index = lo + t;
            ushort kb = 0u;
            ushort vb = 0u;
            if (index < hi) {
                uint row = kv * ctx + index;
                kb = k[row * BLOOMERY_ATTN_HEAD + dim];
                vb = v[row * BLOOMERY_ATTN_HEAD + dim];
            }
            tile_k[e] = kb;
            tile_v[e] = vb;
        }
    }

    float key(uint t, uint, uint dim) { return attn_half(tile_k[t * BLOOMERY_ATTN_HEAD + dim]); }

    float value(uint t, uint, uint dim) { return attn_half(tile_v[t * BLOOMERY_ATTN_HEAD + dim]); }

    void store_k(threadgroup float *row, uint cache_row, device atomic_uint *, uint) {
        attn_store_f16(row, k, cache_row);
    }

    void store_v(threadgroup float *row, uint cache_row, device atomic_uint *, uint) {
        attn_store_f16(row, v, cache_row);
    }
};

// The q8_0 cache. Codes are 64 words a row, scales 8, low byte first.
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
            uint w = e - t * 64u;
            uint index = lo + t;
            uint kc = 0u;
            uint vc = 0u;
            if (index < hi) {
                uint row = kv * ctx + index;
                kc = k_codes[row * 64u + w];
                vc = v_codes[row * 64u + w];
            }
            tile_k_codes[e] = kc;
            tile_v_codes[e] = vc;
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

    float key(uint t, uint, uint dim) {
        return attn_dequant(tile_k_codes[t * 64u + dim / 4u], tile_k_scales[t * 8u + dim / 32u],
                            dim);
    }

    float value(uint t, uint, uint dim) {
        return attn_dequant(tile_v_codes[t * 64u + dim / 4u], tile_v_scales[t * 8u + dim / 32u],
                            dim);
    }

    void store_k(threadgroup float *row, uint cache_row, device atomic_uint *faults, uint layer) {
        attn_store_q8(row, k_codes, k_scales, cache_row, faults, layer);
    }

    void store_v(threadgroup float *row, uint cache_row, device atomic_uint *faults, uint layer) {
        attn_store_q8(row, v_codes, v_scales, cache_row, faults, layer);
    }
};

// The segment that owns the position appends that key and value. Lane 0 of
// SIMD group 0 writes the row; every lane of the group builds it.
template <typename Cache>
static void attn_append(Cache cache,
                        constant AttnArgs &args,
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
        uint i = dim0 + d;
        row0[i] = k[kv * BLOOMERY_ATTN_HEAD + i];
    }
    float y[8];
    attn_norm_rope(row0, k_gain, rope_row, lane, args.eps, y);
    for (uint d = 0u; d < 8u; ++d) {
        row0[dim0 + d] = y[d];
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u) {
        cache.store_k(row0, kv * args.ctx + args.pos, faults, args.layer);
    }
    for (uint d = 0u; d < 8u; ++d) {
        uint i = dim0 + d;
        row0[i] = v[kv * BLOOMERY_ATTN_HEAD + i];
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u) {
        cache.store_v(row0, kv * args.ctx + args.pos, faults, args.layer);
    }
}

// This group's two query heads, normed and partially roped, in `qv`.
static void attn_prepare_q(constant AttnArgs &args,
                           device float const *q,
                           device float const *q_gain,
                           device float const *rope,
                           bool ok,
                           uint kv,
                           uint sg,
                           uint lane,
                           threadgroup float *scratch,
                           thread float qv[BLOOMERY_ATTN_HPT][8]) {
    for (uint h = 0u; h < BLOOMERY_ATTN_HPT; ++h) {
        for (uint d = 0u; d < 8u; ++d) {
            qv[h][d] = 0.0f;
        }
    }
    if (!ok) {
        return;
    }
    for (uint h = 0u; h < BLOOMERY_ATTN_HPT; ++h) {
        uint head = kv * BLOOMERY_ATTN_GQA + sg * BLOOMERY_ATTN_HPT + h;
        uint dim0 = lane * 8u;
        threadgroup float *row = scratch + sg * BLOOMERY_ATTN_HEAD;
        for (uint d = 0u; d < 8u; ++d) {
            row[dim0 + d] = q[head * BLOOMERY_ATTN_QG + dim0 + d];
        }
        float y[8];
        attn_norm_rope(row, q_gain, rope + args.pos * BLOOMERY_ATTN_ROT, lane, args.eps, y);
        for (uint d = 0u; d < 8u; ++d) {
            qv[h][d] = y[d];
        }
    }
}

// Online softmax over this segment's tiles. The running max, weight sum and
// values stay in registers.
template <typename Cache>
static void attn_tile_loop(Cache cache,
                           constant AttnArgs &args,
                           bool ok,
                           uint kv,
                           uint seg,
                           uint span,
                           uint max_live,
                           uint tid,
                           uint lane,
                           thread float qv[BLOOMERY_ATTN_HPT][8],
                           thread float running_m[BLOOMERY_ATTN_HPT],
                           thread float running_l[BLOOMERY_ATTN_HPT],
                           thread float acc[BLOOMERY_ATTN_HPT][8]) {
    for (uint h = 0u; h < BLOOMERY_ATTN_HPT; ++h) {
        running_m[h] = -INFINITY;
        running_l[h] = 0.0f;
        for (uint d = 0u; d < 8u; ++d) {
            acc[h][d] = 0.0f;
        }
    }
    uint lo0 = seg * span;
    uint seg_hi = min((seg + 1u) * span, max_live);
    for (uint lo = lo0; lo < seg_hi; lo += BLOOMERY_ATTN_TILE) {
        uint hi = min(lo + BLOOMERY_ATTN_TILE, seg_hi);
        cache.load(tid, kv, args.ctx, lo, hi);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint dim0 = lane * 8u;
        for (uint h = 0u; h < BLOOMERY_ATTN_HPT; ++h) {
            if (!ok) {
                continue;
            }
            float scores[BLOOMERY_ATTN_TILE];
            bool live[BLOOMERY_ATTN_TILE];
            float tile_max = -INFINITY;
            uint nvalid = 0u;
            for (uint t = 0u; t < BLOOMERY_ATTN_TILE; ++t) {
                uint index = lo + t;
                bool on = index < hi && index < args.keys;
                live[t] = on;
                float dot = 0.0f;
                if (on) {
                    for (uint d = 0u; d < 8u; ++d) {
                        float kvv = cache.key(t, index, dim0 + d);
                        dot = fma(qv[h][d], kvv, dot);
                    }
                }
                // `on` is the same for every lane. The sum still runs on the
                // lanes that loaded nothing, whose dot is zero.
                // The score band counts at most 31 adds, whichever tree the
                // language picks, so this is the built-in reduction.
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
            float new_m = max(running_m[h], tile_max);
            float old_m = running_m[h];
            float factor = old_m > -INFINITY ? exp(old_m - new_m) : 0.0f;
            running_l[h] *= factor;
            for (uint d = 0u; d < 8u; ++d) {
                acc[h][d] *= factor;
            }
            running_m[h] = new_m;
            for (uint t = 0u; t < BLOOMERY_ATTN_TILE; ++t) {
                if (!live[t]) {
                    continue;
                }
                uint index = lo + t;
                float weight = exp(scores[t] - new_m);
                running_l[h] += weight;
                for (uint d = 0u; d < 8u; ++d) {
                    float vv = cache.value(t, index, dim0 + d);
                    acc[h][d] = fma(weight, vv, acc[h][d]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// One segment writes the gated output. More than one writes the partial.
static void attn_epilogue(constant AttnArgs &args,
                          device float const *q,
                          device float *partials,
                          device float *out,
                          device atomic_uint *faults,
                          bool ok,
                          uint kv,
                          uint seg,
                          uint sg,
                          uint lane,
                          thread float running_m[BLOOMERY_ATTN_HPT],
                          thread float running_l[BLOOMERY_ATTN_HPT],
                          thread float acc[BLOOMERY_ATTN_HPT][8]) {
    uint dim0 = lane * 8u;
    for (uint h = 0u; h < BLOOMERY_ATTN_HPT; ++h) {
        uint head = kv * BLOOMERY_ATTN_GQA + sg * BLOOMERY_ATTN_HPT + h;
        if (args.segments == 1u) {
            uint o = head * BLOOMERY_ATTN_HEAD + dim0;
            if (!ok || !(running_l[h] > 0.0f) || !isfinite(running_m[h])) {
                for (uint d = 0u; d < 8u; ++d) {
                    out[o + d] = NAN;
                }
                continue;
            }
            float inv = 1.0f / running_l[h];
            for (uint d = 0u; d < 8u; ++d) {
                float attn = acc[h][d] * inv;
                float g = q[head * BLOOMERY_ATTN_QG + BLOOMERY_ATTN_HEAD + dim0 + d];
                float gated = attn * attn_sigmoid(g);
                attn_commit(out, o + d, attn, g, gated, faults, args.layer);
            }
        } else {
            uint base = (head * args.segments + seg) * BLOOMERY_ATTN_PARTIAL;
            if (!ok) {
                if (lane == 0u) {
                    partials[base] = NAN;
                    partials[base + 1u] = NAN;
                }
                for (uint d = 0u; d < 8u; ++d) {
                    partials[base + 2u + dim0 + d] = NAN;
                }
                continue;
            }
            if (lane == 0u) {
                partials[base] = running_m[h];
                partials[base + 1u] = running_l[h];
            }
            for (uint d = 0u; d < 8u; ++d) {
                partials[base + 2u + dim0 + d] = acc[h][d];
            }
        }
    }
}

template <typename Cache>
static void attn_flash(Cache cache,
                       constant AttnArgs &args,
                       device float const *q,
                       device float const *k,
                       device float const *v,
                       device float const *q_gain,
                       device float const *k_gain,
                       device float const *rope,
                       device float *partials,
                       device float *out,
                       device atomic_uint *faults,
                       uint group,
                       uint tid,
                       threadgroup float *scratch) {
    uint sg = tid / BLOOMERY_ATTN_LANES;
    uint lane = tid % BLOOMERY_ATTN_LANES;
    uint kv = group / args.segments;
    uint seg = group - kv * args.segments;
    if (kv >= BLOOMERY_ATTN_KV || seg >= args.segments) {
        return;
    }

    bool key_bad = args.keys == 0u || args.keys > args.ctx;
    bool pos_bad = args.pos >= args.ctx;
    bool ok = !key_bad && !pos_bad;
    uint max_live = key_bad ? 0u : args.keys;
    if (tid == 0u) {
        if (key_bad) {
            raise(faults, args.layer, BLOOMERY_FAULT_SITE_KEY_COUNT);
        }
        if (pos_bad) {
            raise(faults, args.layer, BLOOMERY_FAULT_SITE_CACHE_POS);
        }
    }

    uint span = attn_span(max_live, args.segments);
    attn_append(cache, args, k, v, k_gain, rope, faults, ok, kv, seg, span, sg, lane, scratch);
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

    float qv[BLOOMERY_ATTN_HPT][8];
    attn_prepare_q(args, q, q_gain, rope, ok, kv, sg, lane, scratch, qv);

    float running_m[BLOOMERY_ATTN_HPT];
    float running_l[BLOOMERY_ATTN_HPT];
    float acc[BLOOMERY_ATTN_HPT][8];
    attn_tile_loop(cache, args, ok, kv, seg, span, max_live, tid, lane, qv, running_m, running_l,
                   acc);
    attn_epilogue(args, q, partials, out, faults, ok, kv, seg, sg, lane, running_m, running_l, acc);
}

kernel void gqa256_flash_f16(device float const *q [[buffer(0)]],
                             device float const *k [[buffer(1)]],
                             device float const *v [[buffer(2)]],
                             device float const *q_gain [[buffer(3)]],
                             device float const *k_gain [[buffer(4)]],
                             device float const *rope [[buffer(5)]],
                             device ushort *cache_k [[buffer(6)]],
                             device ushort *cache_v [[buffer(7)]],
                             device float *partials [[buffer(8)]],
                             device float *out [[buffer(9)]],
                             constant AttnArgs &args [[buffer(10)]],
                             device atomic_uint *faults [[buffer(11)]],
                             uint tid [[thread_index_in_threadgroup]],
                             uint3 group [[threadgroup_position_in_grid]]) {
    threadgroup float scratch[BLOOMERY_ATTN_SG * BLOOMERY_ATTN_HEAD];
    threadgroup ushort tile_k[BLOOMERY_ATTN_TILE * BLOOMERY_ATTN_HEAD];
    threadgroup ushort tile_v[BLOOMERY_ATTN_TILE * BLOOMERY_ATTN_HEAD];
    F16Cache cache;
    cache.k = cache_k;
    cache.v = cache_v;
    cache.tile_k = tile_k;
    cache.tile_v = tile_v;
    attn_flash(cache, args, q, k, v, q_gain, k_gain, rope, partials, out, faults, group.x, tid,
               scratch);
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
                            device float *partials [[buffer(10)]],
                            device float *out [[buffer(11)]],
                            constant AttnArgs &args [[buffer(12)]],
                            device atomic_uint *faults [[buffer(13)]],
                            uint tid [[thread_index_in_threadgroup]],
                            uint3 group [[threadgroup_position_in_grid]]) {
    threadgroup float scratch[BLOOMERY_ATTN_SG * BLOOMERY_ATTN_HEAD];
    threadgroup uint tile_k_codes[BLOOMERY_ATTN_TILE * 64];
    threadgroup ushort tile_k_scales[BLOOMERY_ATTN_TILE * 8];
    threadgroup uint tile_v_codes[BLOOMERY_ATTN_TILE * 64];
    threadgroup ushort tile_v_scales[BLOOMERY_ATTN_TILE * 8];
    Q8Cache cache;
    cache.k_codes = k_codes;
    cache.k_scales = k_scales;
    cache.v_codes = v_codes;
    cache.v_scales = v_scales;
    cache.tile_k_codes = tile_k_codes;
    cache.tile_k_scales = tile_k_scales;
    cache.tile_v_codes = tile_v_codes;
    cache.tile_v_scales = tile_v_scales;
    attn_flash(cache, args, q, k, v, q_gain, k_gain, rope, partials, out, faults, group.x, tid,
               scratch);
}

// Fold one head's segment partials in ascending segment order, apply the
// gate and write the output. A NaN partial (a refused position) writes NaN
// and stops. An empty partial is skipped.
kernel void gqa256_merge(device float const *partials [[buffer(0)]],
                         device float *out [[buffer(1)]],
                         device float const *q [[buffer(2)]],
                         constant AttnArgs &args [[buffer(3)]],
                         device atomic_uint *faults [[buffer(4)]],
                         uint tid [[thread_index_in_threadgroup]],
                         uint3 group [[threadgroup_position_in_grid]]) {
    uint head = group.x;
    if (head >= BLOOMERY_ATTN_Q) {
        return;
    }
    uint lane = tid;
    uint dim0 = lane * 8u;
    float m = -INFINITY;
    float l = 0.0f;
    float acc[8];
    for (uint d = 0u; d < 8u; ++d) {
        acc[d] = 0.0f;
    }
    bool refused = false;
    for (uint seg = 0u; seg < args.segments; ++seg) {
        uint base = (head * args.segments + seg) * BLOOMERY_ATTN_PARTIAL;
        float mj = partials[base];
        float lj = partials[base + 1u];
        if (isnan(mj) || isnan(lj)) {
            refused = true;
            break;
        }
        if (!(lj > 0.0f)) {
            continue;
        }
        float vj[8];
        for (uint d = 0u; d < 8u; ++d) {
            vj[d] = partials[base + 2u + dim0 + d];
        }
        if (mj > m) {
            float f = m > -INFINITY ? exp(m - mj) : 0.0f;
            m = mj;
            l = fma(l, f, lj);
            for (uint d = 0u; d < 8u; ++d) {
                acc[d] = fma(acc[d], f, vj[d]);
            }
        } else {
            float g = exp(mj - m);
            l = fma(lj, g, l);
            for (uint d = 0u; d < 8u; ++d) {
                acc[d] = fma(vj[d], g, acc[d]);
            }
        }
    }
    uint o = head * BLOOMERY_ATTN_HEAD + dim0;
    if (refused || !(l > 0.0f) || !isfinite(m)) {
        for (uint d = 0u; d < 8u; ++d) {
            out[o + d] = NAN;
        }
        return;
    }
    float inv = 1.0f / l;
    for (uint d = 0u; d < 8u; ++d) {
        float attn = acc[d] * inv;
        float gate = q[head * BLOOMERY_ATTN_QG + BLOOMERY_ATTN_HEAD + dim0 + d];
        float merged = attn * attn_sigmoid(gate);
        attn_commit(out, o + d, attn, gate, merged, faults, args.layer);
    }
}

// Every f16 bit pattern as f32, one value per thread. The host compares it
// with `half_to_f32`.
kernel void gqa256_f16_expand(device float *out [[buffer(0)]],
                              uint gid [[thread_position_in_grid]]) {
    if (gid >= 65536u) {
        return;
    }
    out[gid] = attn_half(ushort(gid));
}
