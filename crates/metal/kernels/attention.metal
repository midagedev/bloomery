// Decode attention for one grouped-query layer: 16 query heads, 2 key heads,
// head dim 256. A threadgroup is one (key head, segment). It holds one tile
// of 16 K/V rows in threadgroup memory and four SIMD groups (two query heads
// each) read that tile.
//
// attn_reread is this kernel's value of function constant 0. The combined
// source declares that constant once, in norm.metal, as
// BLOOMERY_MATVEC_SIMD_GROUPS; a second function_constant(0) does not
// compile. The host specializes this kernel with 0, the threadgroup tile
// (the shipped default), or 1, which re-fetches each K/V value from the
// cache once per SIMD group: gqa_factor / HPT = 8 / 2 = 4. A mat-vec's
// specialization of the same index is its own. Both arms run the same
// arithmetic.
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
// Split-K writes one partial per segment. gqa256_merge folds them in segment
// order with the online-softmax fold (the smaller max rescaled) and writes
// the attention. queries is 1 or 2; both sit in the same tile. There is no
// second attention body.

#pragma METAL fp contract(off)

#define attn_reread BLOOMERY_MATVEC_SIMD_GROUPS

#define ATTN_HEAD 256u
#define ATTN_ROT 64u
#define ATTN_Q 16u
#define ATTN_KV 2u
#define ATTN_TILE 16u
#define ATTN_SG 4u
#define ATTN_HPT 2u
#define ATTN_GQA 8u
#define ATTN_PARTIAL 258u

struct AttnArgs {
    uint queries;
    uint segments;
    uint ctx;
    uint layer;
    float eps;
    float scale;
    uint pos0;
    uint pos1;
    uint keys0;
    uint keys1;
};

static_assert(sizeof(AttnArgs) == 40, "AttnArgs is 40 bytes: the host packs ten words");

static uint attn_pos(constant AttnArgs &args, uint q) {
    return q == 0u ? args.pos0 : args.pos1;
}

static uint attn_keys(constant AttnArgs &args, uint q) {
    return q == 0u ? args.keys0 : args.keys1;
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

// f16 bits to f32, exact, the host `half_to_f32`.
static float attn_f16_to_f32(ushort bits) {
    uint sign = uint(bits >> 15) << 31;
    uint exp = uint((bits >> 10) & 0x1fu);
    uint mant = uint(bits & 0x3ffu);
    uint mag;
    if (exp == 0x1fu) {
        mag = 0x7f800000u | (mant << 13);
    } else if (exp == 0u) {
        if (mant == 0u) {
            mag = 0u;
        } else {
            int e = 127 - 14;
            uint m = mant;
            while ((m & 0x400u) == 0u) {
                m <<= 1u;
                e -= 1;
            }
            mag = (uint(e) << 23) | ((m & 0x3ffu) << 13);
        }
    } else {
        mag = ((exp + 112u) << 23) | (mant << 13);
    }
    return as_type<float>(sign | mag);
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
    return float(code) * attn_f16_to_f32(scale);
}

// Sum of squares in the partial-append pair tree, f32. Every lane returns
// the same sum. `scratch` holds the 256 values.
static float attn_sum_squares(threadgroup float *scratch, uint lane) {
    float warp_acc[4];
    for (uint w = 0u; w < 4u; ++w) {
        uint tid = w * 32u + lane;
        uint i0;
        uint i1;
        if (tid < 32u) {
            i0 = tid;
            i1 = tid + 32u;
        } else {
            i0 = tid + 32u;
            i1 = tid + 128u;
        }
        float s0 = scratch[i0] * scratch[i0];
        float s1 = scratch[i1] * scratch[i1];
        warp_acc[w] = s0 + s1;
    }
    for (uint mask = 16u; mask >= 1u; mask >>= 1u) {
        for (uint w = 0u; w < 4u; ++w) {
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
    float mean = sum / float(ATTN_HEAD);
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
    for (uint d = 0u; d < 8u; ++d) {
        uint i = dim0 + d;
        if (i < 32u) {
            float n0 = scratch[i];
            float n1 = scratch[i + 32u];
            float c = rope_row[2u * i];
            float s = rope_row[2u * i + 1u];
            y[d] = fma(n0, c, -(n1 * s));
        } else if (i < ATTN_ROT) {
            float n0 = scratch[i - 32u];
            float n1 = scratch[i];
            float c = rope_row[2u * (i - 32u)];
            float s = rope_row[2u * (i - 32u) + 1u];
            y[d] = fma(n0, s, n1 * c);
        } else {
            y[d] = scratch[i];
        }
    }
}

static void attn_store_f16(threadgroup float *row,
                           device ushort *cache,
                           uint cache_row,
                           device atomic_uint *faults,
                           uint layer) {
    for (uint i = 0u; i < ATTN_HEAD; ++i) {
        cache[cache_row * ATTN_HEAD + i] = attn_f32_to_f16(row[i]);
    }
    (void)faults;
    (void)layer;
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
        return ATTN_TILE;
    }
    uint per = (live + segments - 1u) / segments;
    per = ((per + ATTN_TILE - 1u) / ATTN_TILE) * ATTN_TILE;
    return per < ATTN_TILE ? ATTN_TILE : per;
}

// Load this lane's value of key `index`, dim `dim`, from the tile or the cache.
template <int KIND>
static float attn_k(uint reread,
                    uint t,
                    uint index,
                    uint dim,
                    uint kv,
                    uint ctx,
                    threadgroup ushort *tile_f16,
                    threadgroup uint *tile_codes,
                    threadgroup ushort *tile_scales,
                    device ushort const *cache_f16,
                    device uint const *codes,
                    device ushort const *scales) {
    if (KIND == 0) {
        ushort bits = reread != 0u ? cache_f16[(kv * ctx + index) * ATTN_HEAD + dim]
                                   : tile_f16[t * ATTN_HEAD + dim];
        return attn_f16_to_f32(bits);
    }
    uint word;
    ushort scale;
    if (reread != 0u) {
        uint row = kv * ctx + index;
        word = codes[row * 64u + dim / 4u];
        scale = scales[row * 8u + dim / 32u];
    } else {
        word = tile_codes[t * 64u + dim / 4u];
        scale = tile_scales[t * 8u + dim / 32u];
    }
    return attn_dequant(word, scale, dim);
}

template <int KIND>
static float attn_v(uint reread,
                    uint t,
                    uint index,
                    uint dim,
                    uint kv,
                    uint ctx,
                    threadgroup ushort *tile_f16,
                    threadgroup uint *tile_codes,
                    threadgroup ushort *tile_scales,
                    device ushort const *cache_f16,
                    device uint const *codes,
                    device ushort const *scales) {
    return attn_k<KIND>(reread, t, index, dim, kv, ctx, tile_f16, tile_codes, tile_scales,
                        cache_f16, codes, scales);
}

template <int KIND>
static void attn_flash(constant AttnArgs &args,
                       device float const *q,
                       device float const *k,
                       device float const *v,
                       device float const *q_gain,
                       device float const *k_gain,
                       device float const *rope,
                       device ushort *cache_k_f16,
                       device ushort *cache_v_f16,
                       device uint *k_codes,
                       device ushort *k_scales,
                       device uint *v_codes,
                       device ushort *v_scales,
                       device float *partials,
                       device atomic_uint *faults,
                       uint reread,
                       uint group,
                       uint tid,
                       threadgroup float *scratch,
                       threadgroup ushort *tile_k_f16,
                       threadgroup ushort *tile_v_f16,
                       threadgroup uint *tile_k_codes,
                       threadgroup ushort *tile_k_scales,
                       threadgroup uint *tile_v_codes,
                       threadgroup ushort *tile_v_scales) {
    uint sg = tid / 32u;
    uint lane = tid % 32u;
    uint kv = group / args.segments;
    uint seg = group - kv * args.segments;
    if (kv >= ATTN_KV || seg >= args.segments) {
        return;
    }

    bool ok[2];
    bool key_bad[2];
    bool pos_bad[2];
    uint max_live = 0u;
    for (uint query = 0u; query < 2u; ++query) {
        ok[query] = false;
        key_bad[query] = false;
        pos_bad[query] = false;
        if (query >= args.queries) {
            continue;
        }
        uint keys = attn_keys(args, query);
        uint pos = attn_pos(args, query);
        key_bad[query] = keys == 0u || keys > args.ctx;
        pos_bad[query] = pos >= args.ctx;
        ok[query] = !key_bad[query] && !pos_bad[query];
        if (!key_bad[query] && keys > max_live) {
            max_live = keys;
        }
    }
    if (tid == 0u) {
        for (uint query = 0u; query < args.queries; ++query) {
            if (key_bad[query]) {
                raise(faults, args.layer, BLOOMERY_FAULT_SITE_KEY_COUNT);
            }
            if (pos_bad[query]) {
                raise(faults, args.layer, BLOOMERY_FAULT_SITE_CACHE_POS);
            }
        }
    }

    uint span = attn_span(max_live, args.segments);
    threadgroup float *row0 = scratch;

    // The segment that owns a query's position appends that key and value.
    // Lane 0 of SIMD group 0 writes the row; every lane of the group builds it.
    if (sg == 0u) {
        for (uint query = 0u; query < args.queries; ++query) {
            uint pos = attn_pos(args, query);
            uint keys = attn_keys(args, query);
            bool owns = ok[query] && pos < keys && pos / span == seg;
            if (!owns) {
                continue;
            }
            uint dim0 = lane * 8u;
            device float const *rope_row = rope + pos * ATTN_ROT;
            for (uint d = 0u; d < 8u; ++d) {
                uint i = dim0 + d;
                row0[i] = k[(query * ATTN_KV + kv) * ATTN_HEAD + i];
            }
            float y[8];
            attn_norm_rope(row0, k_gain, rope_row, lane, args.eps, y);
            for (uint d = 0u; d < 8u; ++d) {
                row0[dim0 + d] = y[d];
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            if (lane == 0u) {
                uint row = kv * args.ctx + pos;
                if (KIND == 0) {
                    attn_store_f16(row0, cache_k_f16, row, faults, args.layer);
                } else {
                    attn_store_q8(row0, k_codes, k_scales, row, faults, args.layer);
                }
            }
            for (uint d = 0u; d < 8u; ++d) {
                uint i = dim0 + d;
                row0[i] = v[(query * ATTN_KV + kv) * ATTN_HEAD + i];
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            if (lane == 0u) {
                uint row = kv * args.ctx + pos;
                if (KIND == 0) {
                    attn_store_f16(row0, cache_v_f16, row, faults, args.layer);
                } else {
                    attn_store_q8(row0, v_codes, v_scales, row, faults, args.layer);
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

    // This group's two query heads, for every query the launch carries.
    float qv[2][2][8];
    for (uint h = 0u; h < ATTN_HPT; ++h) {
        for (uint query = 0u; query < 2u; ++query) {
            for (uint d = 0u; d < 8u; ++d) {
                qv[h][query][d] = 0.0f;
            }
        }
    }
    for (uint h = 0u; h < ATTN_HPT; ++h) {
        uint head = kv * ATTN_GQA + sg * ATTN_HPT + h;
        for (uint query = 0u; query < args.queries; ++query) {
            if (!ok[query]) {
                continue;
            }
            uint pos = attn_pos(args, query);
            uint dim0 = lane * 8u;
            threadgroup float *row = scratch + sg * ATTN_HEAD;
            for (uint d = 0u; d < 8u; ++d) {
                uint i = dim0 + d;
                row[i] = q[(query * ATTN_Q + head) * ATTN_HEAD + i];
            }
            float y[8];
            attn_norm_rope(row, q_gain, rope + pos * ATTN_ROT, lane, args.eps, y);
            for (uint d = 0u; d < 8u; ++d) {
                qv[h][query][d] = y[d];
            }
        }
    }

    float running_m[2][2];
    float running_l[2][2];
    float acc[2][2][8];
    for (uint h = 0u; h < ATTN_HPT; ++h) {
        for (uint query = 0u; query < 2u; ++query) {
            running_m[h][query] = -INFINITY;
            running_l[h][query] = 0.0f;
            for (uint d = 0u; d < 8u; ++d) {
                acc[h][query][d] = 0.0f;
            }
        }
    }

    // A segment owns [seg * span, min((seg + 1) * span, max_live)). The
    // next segment starts where this one ends, so a key is in one partial.
    uint lo0 = seg * span;
    uint seg_hi = min((seg + 1u) * span, max_live);
    for (uint lo = lo0; lo < seg_hi; lo += ATTN_TILE) {
        uint hi = min(lo + ATTN_TILE, seg_hi);
        if (reread == 0u) {
            if (KIND == 0) {
                for (uint e = tid; e < ATTN_TILE * ATTN_HEAD; e += 128u) {
                    uint t = e / ATTN_HEAD;
                    uint dim = e - t * ATTN_HEAD;
                    uint index = lo + t;
                    ushort kb = 0u;
                    ushort vb = 0u;
                    if (index < hi) {
                        uint row = kv * args.ctx + index;
                        kb = cache_k_f16[row * ATTN_HEAD + dim];
                        vb = cache_v_f16[row * ATTN_HEAD + dim];
                    }
                    tile_k_f16[e] = kb;
                    tile_v_f16[e] = vb;
                }
            } else {
                for (uint e = tid; e < ATTN_TILE * 64u; e += 128u) {
                    uint t = e / 64u;
                    uint w = e - t * 64u;
                    uint index = lo + t;
                    uint kc = 0u;
                    uint vc = 0u;
                    if (index < hi) {
                        uint row = kv * args.ctx + index;
                        kc = k_codes[row * 64u + w];
                        vc = v_codes[row * 64u + w];
                    }
                    tile_k_codes[e] = kc;
                    tile_v_codes[e] = vc;
                }
                if (tid < ATTN_TILE * 8u) {
                    uint t = tid / 8u;
                    uint b = tid - t * 8u;
                    uint index = lo + t;
                    ushort ks = 0u;
                    ushort vs = 0u;
                    if (index < hi) {
                        uint row = kv * args.ctx + index;
                        ks = k_scales[row * 8u + b];
                        vs = v_scales[row * 8u + b];
                    }
                    tile_k_scales[tid] = ks;
                    tile_v_scales[tid] = vs;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint dim0 = lane * 8u;
        for (uint h = 0u; h < ATTN_HPT; ++h) {
            for (uint query = 0u; query < args.queries; ++query) {
                if (!ok[query]) {
                    continue;
                }
                uint keys = attn_keys(args, query);
                float scores[ATTN_TILE];
                bool live[ATTN_TILE];
                float tile_max = -INFINITY;
                uint nvalid = 0u;
                for (uint t = 0u; t < ATTN_TILE; ++t) {
                    uint index = lo + t;
                    bool on = index < hi && index < keys;
                    live[t] = on;
                    float dot = 0.0f;
                    if (on) {
                        for (uint d = 0u; d < 8u; ++d) {
                            float kvv = attn_k<KIND>(reread, t, index, dim0 + d, kv, args.ctx,
                                                     tile_k_f16, tile_k_codes, tile_k_scales,
                                                     cache_k_f16, k_codes, k_scales);
                            dot = fma(qv[h][query][d], kvv, dot);
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
                float new_m = max(running_m[h][query], tile_max);
                float old_m = running_m[h][query];
                float factor = old_m > -INFINITY ? exp(old_m - new_m) : 0.0f;
                running_l[h][query] *= factor;
                for (uint d = 0u; d < 8u; ++d) {
                    acc[h][query][d] *= factor;
                }
                running_m[h][query] = new_m;
                for (uint t = 0u; t < ATTN_TILE; ++t) {
                    if (!live[t]) {
                        continue;
                    }
                    uint index = lo + t;
                    float weight = exp(scores[t] - new_m);
                    running_l[h][query] += weight;
                    for (uint d = 0u; d < 8u; ++d) {
                        float vv = attn_v<KIND>(reread, t, index, dim0 + d, kv, args.ctx,
                                                tile_v_f16, tile_v_codes, tile_v_scales,
                                                cache_v_f16, v_codes, v_scales);
                        acc[h][query][d] = fma(weight, vv, acc[h][query][d]);
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint dim0 = lane * 8u;
    for (uint h = 0u; h < ATTN_HPT; ++h) {
        uint head = kv * ATTN_GQA + sg * ATTN_HPT + h;
        for (uint query = 0u; query < args.queries; ++query) {
            uint base = ((query * ATTN_Q + head) * args.segments + seg) * ATTN_PARTIAL;
            if (!ok[query]) {
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
                partials[base] = running_m[h][query];
                partials[base + 1u] = running_l[h][query];
            }
            for (uint d = 0u; d < 8u; ++d) {
                partials[base + 2u + dim0 + d] = acc[h][query][d];
            }
        }
    }
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
    (void)out;
    threadgroup float scratch[ATTN_SG * ATTN_HEAD];
    threadgroup ushort tile_k_f16[ATTN_TILE * ATTN_HEAD];
    threadgroup ushort tile_v_f16[ATTN_TILE * ATTN_HEAD];
    threadgroup uint tile_k_codes[1];
    threadgroup ushort tile_k_scales[1];
    threadgroup uint tile_v_codes[1];
    threadgroup ushort tile_v_scales[1];
    attn_flash<0>(args, q, k, v, q_gain, k_gain, rope, cache_k, cache_v,
                  reinterpret_cast<device uint *>(cache_k),
                  reinterpret_cast<device ushort *>(cache_k),
                  reinterpret_cast<device uint *>(cache_v),
                  reinterpret_cast<device ushort *>(cache_v),
                  partials, faults, attn_reread, group.x, tid, scratch, tile_k_f16, tile_v_f16,
                  tile_k_codes, tile_k_scales, tile_v_codes, tile_v_scales);
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
    (void)out;
    threadgroup float scratch[ATTN_SG * ATTN_HEAD];
    threadgroup ushort tile_k_f16[1];
    threadgroup ushort tile_v_f16[1];
    threadgroup uint tile_k_codes[ATTN_TILE * 64];
    threadgroup ushort tile_k_scales[ATTN_TILE * 8];
    threadgroup uint tile_v_codes[ATTN_TILE * 64];
    threadgroup ushort tile_v_scales[ATTN_TILE * 8];
    attn_flash<1>(args, q, k, v, q_gain, k_gain, rope, k_scales, v_scales, k_codes, k_scales,
                  v_codes, v_scales, partials, faults, attn_reread, group.x, tid, scratch,
                  tile_k_f16, tile_v_f16, tile_k_codes, tile_k_scales, tile_v_codes, tile_v_scales);
}

// Fold one head's segment partials in ascending segment order and write the
// attention. A NaN partial (a refused query) writes NaN and stops.
kernel void gqa256_merge(device float const *partials [[buffer(0)]],
                         device float *out [[buffer(1)]],
                         constant AttnArgs &args [[buffer(2)]],
                         uint tid [[thread_index_in_threadgroup]],
                         uint3 group [[threadgroup_position_in_grid]]) {
    uint query = group.x / ATTN_Q;
    uint head = group.x - query * ATTN_Q;
    if (query >= args.queries || head >= ATTN_Q) {
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
        uint base = ((query * ATTN_Q + head) * args.segments + seg) * ATTN_PARTIAL;
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
    uint o = (query * ATTN_Q + head) * ATTN_HEAD + dim0;
    if (refused || !(l > 0.0f) || !isfinite(m)) {
        for (uint d = 0u; d < 8u; ++d) {
            out[o + d] = NAN;
        }
        return;
    }
    float inv = 1.0f / l;
    for (uint d = 0u; d < 8u; ++d) {
        out[o + d] = acc[d] * inv;
    }
}
