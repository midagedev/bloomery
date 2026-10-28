// Decode attention for one grouped-query layer at the instance's geometry:
// BLOOMERY_ATTN_Q query heads over BLOOMERY_ATTN_KV key heads, BLOOMERY_ATTN_GQA
// a key head, BLOOMERY_ATTN_HEAD values a head. A threadgroup is one (key head,
// segment) of BLOOMERY_ATTN_GQA SIMD groups; each SIMD group holds one query head, and
// its 32 lanes split the head, BLOOMERY_ATTN_LANE_DIMS dims a lane (8 at
// head 256, 4 at head 128). The segment's keys stream from the cache
// straight into registers, one lane word a key a lane (16 bytes at 8 dims,
// 8 at 4).
//
// The segment that owns the position appends that key and value. The whole
// owning SIMD group builds each row: the K row is normed and roped through
// threadgroup scratch, and every lane converts and stores its own values,
// so the row lands in one pass instead of one lane's serial stores. The
// q8_0 append shares the row out a lane group a 32-value block (a quad at 8
// dims a lane, an octet at 4).
//
// One segment is the flash alone: the epilogue normalizes, applies the gate
// when the layer has one, and writes the output. More than one writes a
// partial per segment, and the last threadgroup of the dispatch — the
// ticket's — folds those partials, applies the gate and writes the output,
// in the merge's arithmetic, then resets the ticket. There is no merge
// dispatch.
//
// Numeric order of the norm and the rope, from rope_neox's append over a
// head of HEAD values whose first ROT turn: thread t of the HEAD/2 owns the
// square pair `owned_pair(HEAD, ROT, t)`, (t, t + ROT/2) for t < ROT/2 and
// (t + ROT/2, t + HEAD/2) otherwise, over HEAD/64 warps. At head 256 the
// rope turns the first 64 values and the other 192 pass through; at head
// 128 every value turns. The CUDA sum is f64; this GPU has no f64, so the
// same pairing, the same xor butterfly and the same warp order are f32.
// The rope pair is that kernel's fma form: y0 = fma(n0, c, -(n1 * s)),
// y1 = fma(n0, s, n1 * c), table row pos * ROT, pair i at 2i, 2i + 1.
// A q8_0 append is gguf quantize_q8_0: scale f16(amax / 127), codes
// nearest_int(v * id) with id = 127 / amax (0 when amax is 0 or id
// overflows).
//
// The score is each lane's fmas in dim order, then the SIMD group's
// built-in sum (at most 31 adds, whichever tree the language picks), then
// the scale (exact at head 256, ik's f32 1/sqrt(head) otherwise), as the
// band in the oracle derives. The softmax rescales once a 16-key tile; the
// fold adds the segments' partials in ascending segment order. The value
// sum is one fma a key in key order, per dim.
//
// A tile loop holds no branch on a key's liveness: a key past the segment's
// end is a zero row, scores -infinity and weighs exactly 0, so the tile's
// sixteen keys are one straight line the compiler schedules together. The
// next tile's rows are fetched into registers before the arithmetic on this
// tile and stored to threadgroup memory after it.
//
// The rows launches run a chunk of causal queries. attn_norm_rope_rows appends
// row r's key and value at position from + r, by the same append the decode
// flash runs, one SIMD group a (key head, row). gqa256_flash_f16_rows is the
// decode flash with a row grid dimension: row r is the launch at position
// from + r over from + r + 1 keys, with its own partials and ticket word, and
// it reads the rows the append dispatch wrote. A row is the bits of the decode
// launch at its position with the same segment count.
//
// The prefill kernel (attn_prefill_f16, at the end of the file) is the same
// attention over a chunk of rows for the prompt pass: simdgroup matrices over
// tiles of keys, f16 operands. It reads the cache the append dispatch wrote
// and is held to the decode flash by a band.
//
// A gated layer's gate (BLOOMERY_ATTN_GATED) is the second half of each
// query head, the raw projection. The output value is
// attn * (1 / (1 + exp(-g))), after the softmax. A non-finite gate, or a
// non-finite product of a finite attention, raises the product site and
// stores NaN: a non-finite gate can otherwise leave the attention itself,
// which is a plausible value. A non-finite attention stores NaN and raises
// nothing, gated or not. That fault is already the quant site, and the
// product site is the smaller code, so raising it would replace the quant
// site.
//
// A per-head gate (BLOOMERY_ATTN_HEAD_GATE) is one raw value a query head from
// a buffer of its own, BLOOMERY_ATTN_Q values a row; every value of the head
// is multiplied by its sigmoid, under the same fault rules as the packed gate.
//
// A layer's window (BLOOMERY_ATTN_WINDOW, the launch word `window`, 0 for
// none) is the last `window` of its live keys: of `keys` live keys, key k is
// attended when k + window >= keys. The segments tile those keys alone, so a
// key before the window is never fetched, and a segment the caller counted
// that holds none of them is empty and skipped by the fold. An append position
// before the window belongs to the first segment. The prefill kernel starts at
// the tile of the first window of its block and masks every row's own.

#pragma METAL fp contract(off)

static_assert(BLOOMERY_ATTN_GQA == BLOOMERY_ATTN_Q / BLOOMERY_ATTN_KV,
              "a key head's query heads are the group");
static_assert(BLOOMERY_ATTN_HPT * BLOOMERY_ATTN_SG == BLOOMERY_ATTN_GQA,
              "the SIMD groups of one head each cover a key head");
static_assert(BLOOMERY_ATTN_PARTIAL == BLOOMERY_ATTN_HEAD + 2u,
              "a partial is the max, the weight sum and the head");
static_assert(BLOOMERY_ATTN_GATED == 0u || BLOOMERY_ATTN_GATED == 1u,
              "a layer's query is gated or plain");
static_assert(BLOOMERY_ATTN_QG == BLOOMERY_ATTN_HEAD * (1u + BLOOMERY_ATTN_GATED),
              "a query head is the query and then the gate when the layer has one");
static_assert(BLOOMERY_ATTN_HEAD_GATE == 0u || BLOOMERY_ATTN_HEAD_GATE == 1u,
              "a layer's output gate is per head or it is not");
static_assert(BLOOMERY_ATTN_GATED == 0u || BLOOMERY_ATTN_HEAD_GATE == 0u,
              "a layer's output gate is packed in its query or per head, not both");
static_assert(BLOOMERY_ATTN_WINDOW == 0u || BLOOMERY_ATTN_WINDOW == 1u,
              "a flash cuts a window or it does not");
static_assert(BLOOMERY_ATTN_THREADS == BLOOMERY_ATTN_LANES * BLOOMERY_ATTN_SG,
              "the flash threadgroup is the key head's SIMD groups");
static_assert(BLOOMERY_ATTN_LANES * BLOOMERY_ATTN_LANE_DIMS == BLOOMERY_ATTN_HEAD,
              "a lane's values cover the head");
static_assert(BLOOMERY_ATTN_LANE_DIMS == 8u || BLOOMERY_ATTN_LANE_DIMS == 4u,
              "a lane word is 16 or 8 bytes of halves");
static_assert(BLOOMERY_ATTN_LANES % BLOOMERY_ATTN_KV == 0u,
              "the fold's threads a head divide the lanes of a SIMD group");
static_assert(BLOOMERY_ATTN_Q * (BLOOMERY_ATTN_LANES / BLOOMERY_ATTN_KV) == BLOOMERY_ATTN_THREADS,
              "the folding group is a thread to each head's share");
static_assert(BLOOMERY_ATTN_HEAD % 64u == 0u &&
                  (BLOOMERY_ATTN_ROT / 2u <= BLOOMERY_ATTN_LANES ||
                   BLOOMERY_ATTN_ROT == BLOOMERY_ATTN_HEAD),
              "the norm's pair tree: whole warps, the turning pairs in warp 0 or every value");

// A lane's word of halves: its BLOOMERY_ATTN_LANE_DIMS values of a cache row.
#if BLOOMERY_ATTN_LANE_DIMS == 8
typedef uint4 attn_lane_word;
#else
typedef uint2 attn_lane_word;
#endif

// The q8_0 cache's row: code words, scales, the lanes of a 32-value block
// and the code words a lane holds.
#define ATTN_Q8_ROW_WORDS (BLOOMERY_ATTN_HEAD / 4u)
#define ATTN_Q8_ROW_SCALES (BLOOMERY_ATTN_HEAD / 32u)
#define ATTN_Q8_BLOCK_LANES (32u / BLOOMERY_ATTN_LANE_DIMS)
#define ATTN_Q8_LANE_WORDS (BLOOMERY_ATTN_LANE_DIMS / 4u)

// The fold's threads a head, and the values each holds: a thread owns the
// values sub, sub + SUBS, ... of its head, so a head's load or store is a run
// of contiguous words across its threads.
#define ATTN_FOLD_SUBS (BLOOMERY_ATTN_LANES / BLOOMERY_ATTN_KV)
#define ATTN_FOLD_VALS (BLOOMERY_ATTN_HEAD / ATTN_FOLD_SUBS)

// The flash's one shared block, in 16-byte words: the K and V f16 tiles,
// or the fold's factors (two floats a segment a head, at the segment cap,
// then three floats a head), whichever is larger.
#define ATTN_TILE_WORDS (BLOOMERY_ATTN_TILE * BLOOMERY_ATTN_HEAD / 4u)
#define ATTN_FACTOR_WORDS \
    ((BLOOMERY_ATTN_Q * (BLOOMERY_ATTN_SEGMENT_CAP * 2u + 3u) + 3u) / 4u)
#define ATTN_SHARED_WORDS \
    (ATTN_TILE_WORDS > ATTN_FACTOR_WORDS ? ATTN_TILE_WORDS : ATTN_FACTOR_WORDS)

static_assert(ATTN_SHARED_WORDS * 4u >= BLOOMERY_ATTN_Q * BLOOMERY_ATTN_SEGMENT_CAP * 2u,
              "the fold's factors at the segment cap fit the shared block");
static_assert(ATTN_SHARED_WORDS * 16u >=
                  2u * BLOOMERY_ATTN_TILE * BLOOMERY_ATTN_LANES * sizeof(attn_lane_word),
              "the K and V tiles fit the shared block");
static_assert(ATTN_SHARED_WORDS * 4u >= BLOOMERY_ATTN_SG * BLOOMERY_ATTN_HEAD,
              "the scratch of every SIMD group's head fits the shared block");

struct AttnArgs {
    uint segments;
    uint ctx;
    uint layer;
    float eps;
    float scale;
    uint pos;
    uint keys;
#if BLOOMERY_ATTN_WINDOW
    uint window;
#endif
};

#if BLOOMERY_ATTN_WINDOW
static_assert(sizeof(AttnArgs) == 32, "AttnArgs is 32 bytes: the host packs eight words");
#else
static_assert(sizeof(AttnArgs) == 28, "AttnArgs is 28 bytes: the host packs seven words");
#endif

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
#if BLOOMERY_ATTN_WINDOW
    uint window;
#endif
};

#if BLOOMERY_ATTN_WINDOW
static_assert(sizeof(AttnRowsArgs) == 48, "AttnRowsArgs is 48 bytes: the host packs twelve words");
#else
static_assert(sizeof(AttnRowsArgs) == 44, "AttnRowsArgs is 44 bytes: the host packs eleven words");
#endif

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
#if BLOOMERY_ATTN_WINDOW
    a.window = args.window;
#endif
    return a;
}

// The first key a layer attends: the last `window` of its `live` keys, and
// the first of them when the launch names no window (0). A flash with no
// window attends every live key.
inline uint attn_window_lo(thread const AttnArgs &args, uint live) {
#if BLOOMERY_ATTN_WINDOW
    return (args.window != 0u && live > args.window) ? live - args.window : 0u;
#else
    return 0u;
#endif
}

// The layer's output gate, packed in its query or one value a head.
#define ATTN_GATE (BLOOMERY_ATTN_GATED || BLOOMERY_ATTN_HEAD_GATE)

// The raw gate of channel `dim` of query head `head`, from `src`: the head's
// packed gate beside its query row, or the head's one value in its row of the
// per-head gate buffer.
inline float attn_gate_value(device float const *src, uint head, uint dim) {
#if BLOOMERY_ATTN_HEAD_GATE
    return src[head];
#else
    return src[head * BLOOMERY_ATTN_QG + BLOOMERY_ATTN_HEAD + dim];
#endif
}

// The per-head gate buffer's kernel parameter, bound by a launch whose layer
// gates per head, and its row `row`'s start as the flash takes it.
#if BLOOMERY_ATTN_HEAD_GATE
#define ATTN_HGATE_PARAM(n) , device float const *hgate [[buffer(n)]]
#define ATTN_HGATE(row) (hgate + (ulong)(row) * BLOOMERY_ATTN_Q)
#else
#define ATTN_HGATE_PARAM(n)
#define ATTN_HGATE(row) ((device float const *)nullptr)
#endif

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

// Sum of squares in the append's pair tree, f32. Every lane returns the
// same sum. `scratch` holds the head's values. The tree is the CUDA
// append's, over HEAD/2 threads of HEAD/64 warps; one SIMD group's 32 lanes
// evaluate it by walking the warps' pairs themselves, so the warp count
// here is the tree's, not the launch's SIMD-group count.
static float attn_sum_squares(threadgroup float *scratch, uint lane) {
    constexpr uint pair_warps = BLOOMERY_ATTN_HEAD / 64u;
    float warp_acc[pair_warps];
    for (uint w = 0u; w < pair_warps; ++w) {
        uint tid = w * BLOOMERY_ATTN_LANES + lane;
        uint i0;
        uint i1;
        if (tid < BLOOMERY_ATTN_ROT / 2u) {
            i0 = tid;
            i1 = tid + BLOOMERY_ATTN_ROT / 2u;
        } else {
            i0 = tid + BLOOMERY_ATTN_ROT / 2u;
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
    for (uint w = 1u; w < pair_warps; ++w) {
        sum += warp_acc[w];
    }
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

// Norm then rope of the head in `scratch`, into `y` (this lane's dims). The
// gain is HEAD values and the rope row ROT. `lane` owns dims
// lane*LANE_DIMS .. +LANE_DIMS-1.
static void attn_norm_rope(threadgroup float *scratch,
                           device float const *gain,
                           device float const *rope_row,
                           uint lane,
                           float eps,
                           thread float y[BLOOMERY_ATTN_LANE_DIMS]) {
    simdgroup_barrier(mem_flags::mem_threadgroup);
    float scale = attn_rms_scale(attn_sum_squares(scratch, lane), eps);
    uint dim0 = lane * BLOOMERY_ATTN_LANE_DIMS;
    float nloc[BLOOMERY_ATTN_LANE_DIMS];
    for (uint d = 0u; d < BLOOMERY_ATTN_LANE_DIMS; ++d) {
        uint i = dim0 + d;
        nloc[d] = (scale * gain[i]) * scratch[i];
    }
    for (uint d = 0u; d < BLOOMERY_ATTN_LANE_DIMS; ++d) {
        scratch[dim0 + d] = nloc[d];
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    uint pairs = BLOOMERY_ATTN_ROT / 2u;
    for (uint d = 0u; d < BLOOMERY_ATTN_LANE_DIMS; ++d) {
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

// The lane's values of a row, converted and stored: the row's values land
// in one pass over the group's lanes, each value the same `f32_to_f16_bits`
// the host quantizer runs.
static void attn_store_f16_lane(thread float const row[BLOOMERY_ATTN_LANE_DIMS],
                                uint lane,
                                device ushort *cache,
                                uint cache_row) {
    device ushort *dst = cache + cache_row * BLOOMERY_ATTN_HEAD + lane * BLOOMERY_ATTN_LANE_DIMS;
    for (uint d = 0u; d < BLOOMERY_ATTN_LANE_DIMS; ++d) {
        dst[d] = attn_f32_to_f16(row[d]);
    }
}

// One q8_0 row, the block's lane group to a 32-value block (a quad at 8
// dims a lane, an octet at 4): each lane holds its values of the block, the
// group reduces the amax and the finiteness, and each lane quantizes and
// stores its code words. `quantize_q8_0`'s rule; the amax is exact in any
// order, so the group's bits are the host quantizer's. A non-finite block is
// a NaN scale and zero codes, and raises the quant site.
static void attn_store_q8_lanes(thread float const row[BLOOMERY_ATTN_LANE_DIMS],
                                uint lane,
                                device uint *codes,
                                device ushort *scales,
                                uint cache_row,
                                device atomic_uint *faults,
                                uint layer) {
    float amax = 0.0f;
    uint bad = 0u;
    for (uint d = 0u; d < BLOOMERY_ATTN_LANE_DIMS; ++d) {
        if (!isfinite(row[d])) {
            bad = 1u;
        } else {
            amax = max(amax, abs(row[d]));
        }
    }
    // The group's amaxes and bad flags, every lane.
    for (ushort mask = 1u; mask < ATTN_Q8_BLOCK_LANES; mask <<= 1u) {
        amax = max(amax, simd_shuffle_xor(amax, mask));
        bad = max(bad, simd_shuffle_xor(bad, mask));
    }
    uint block = lane / ATTN_Q8_BLOCK_LANES;
    if (bad != 0u) {
        if (lane % ATTN_Q8_BLOCK_LANES == 0u) {
            scales[cache_row * ATTN_Q8_ROW_SCALES + block] = 0x7e00u;
        }
        for (uint w = 0u; w < ATTN_Q8_LANE_WORDS; ++w) {
            codes[cache_row * ATTN_Q8_ROW_WORDS + lane * ATTN_Q8_LANE_WORDS + w] = 0u;
        }
        if (lane % ATTN_Q8_BLOCK_LANES == 0u) {
            raise(faults, layer, BLOOMERY_FAULT_SITE_KV_QUANT);
        }
        return;
    }
    // The host quantizer's multiplier: 127 / amax, else 0.
    float id = (amax != 0.0f) ? 127.0f / amax : 0.0f;
    if (!isfinite(id)) {
        id = 0.0f;
    }
    if (lane % ATTN_Q8_BLOCK_LANES == 0u) {
        scales[cache_row * ATTN_Q8_ROW_SCALES + block] = attn_f32_to_f16(amax / 127.0f);
    }
    for (uint w = 0u; w < ATTN_Q8_LANE_WORDS; ++w) {
        uint word = 0u;
        for (uint byte = 0u; byte < 4u; ++byte) {
            float v = row[w * 4u + byte];
            int code = clamp(attn_nearest_int(v * id), -128, 127);
            word |= (uint(code) & 0xffu) << (8u * byte);
        }
        codes[cache_row * ATTN_Q8_ROW_WORDS + lane * ATTN_Q8_LANE_WORDS + w] = word;
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

// A lane word's halves, widened by the hardware conversion: a finite half
// is exact either way, and a NaN's sign does not change the attention's
// NaN-ness.
#if BLOOMERY_ATTN_LANE_DIMS == 8
static void attn_widen(uint4 w, thread float out[8]) {
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
#else
static void attn_widen(uint2 w, thread float out[4]) {
    float4 f0 = float4(as_type<half4>(w));
    out[0] = f0.x;
    out[1] = f0.y;
    out[2] = f0.z;
    out[3] = f0.w;
}
#endif

// Store `attn` of a layer with no gate: a non-finite attention stores NaN
// and raises nothing.
inline void attn_commit_plain(device float *out, uint index, float attn) {
    out[index] = isfinite(attn) ? attn : NAN;
}

// The f16 cache. One tile of rows sits in threadgroup memory as lane
// words: the group loads it once, cooperatively, and the SIMD groups of the
// key head read it a head at a time. The lane's values of a tile row are one
// word. A row past the live keys loads as zero bits, the value 0.0f. A
// thread's share of a tile is fetched into registers (`fetch`) and stored to
// threadgroup memory (`commit`) in two steps, so the next tile's loads are in
// flight while this tile's arithmetic runs.
#define ATTN_F16_WORDS (BLOOMERY_ATTN_TILE * BLOOMERY_ATTN_LANES)
#define ATTN_F16_PENDING \
    ((ATTN_F16_WORDS + BLOOMERY_ATTN_THREADS - 1u) / BLOOMERY_ATTN_THREADS)

struct F16Cache {
    device ushort *k;
    device ushort *v;
    threadgroup attn_lane_word *tile_k;
    threadgroup attn_lane_word *tile_v;

    struct Pending {
        attn_lane_word k[ATTN_F16_PENDING];
        attn_lane_word v[ATTN_F16_PENDING];
    };

    void fetch(uint tid, uint kv, uint ctx, uint lo, uint hi, thread Pending &pending) {
        for (uint i = 0u; i < ATTN_F16_PENDING; ++i) {
            uint e = tid + i * BLOOMERY_ATTN_THREADS;
            uint t = e / BLOOMERY_ATTN_LANES;
            uint c = e - t * BLOOMERY_ATTN_LANES;
            uint index = lo + t;
            attn_lane_word wk = 0u;
            attn_lane_word wv = 0u;
            if (e < ATTN_F16_WORDS && index < hi) {
                uint row = kv * ctx + index;
                const device attn_lane_word *pk =
                    (const device attn_lane_word *)(k + row * BLOOMERY_ATTN_HEAD);
                const device attn_lane_word *pv =
                    (const device attn_lane_word *)(v + row * BLOOMERY_ATTN_HEAD);
                wk = pk[c];
                wv = pv[c];
            }
            pending.k[i] = wk;
            pending.v[i] = wv;
        }
    }

    void commit(uint tid, thread const Pending &pending) {
        for (uint i = 0u; i < ATTN_F16_PENDING; ++i) {
            uint e = tid + i * BLOOMERY_ATTN_THREADS;
            if (e < ATTN_F16_WORDS) {
                tile_k[e] = pending.k[i];
                tile_v[e] = pending.v[i];
            }
        }
    }

    void key(uint t, uint lane, thread float out[BLOOMERY_ATTN_LANE_DIMS]) {
        attn_widen(tile_k[t * BLOOMERY_ATTN_LANES + lane], out);
    }

    void value(uint t, uint lane, thread float out[BLOOMERY_ATTN_LANE_DIMS]) {
        attn_widen(tile_v[t * BLOOMERY_ATTN_LANES + lane], out);
    }

    void store_k(thread float const row[BLOOMERY_ATTN_LANE_DIMS], uint lane, uint cache_row,
                 device atomic_uint *, uint) {
        attn_store_f16_lane(row, lane, k, cache_row);
    }

    void store_v(thread float const row[BLOOMERY_ATTN_LANE_DIMS], uint lane, uint cache_row,
                 device atomic_uint *, uint) {
        attn_store_f16_lane(row, lane, v, cache_row);
    }
};

// The q8_0 cache, one tile in threadgroup memory: HEAD/4 code words and
// HEAD/32 scales a row, low byte first. A row past the live keys loads as
// zero codes and a zero scale, the value 0.0f. The lane's values sit in
// its code words of one 32-value block. Fetched and committed as the f16
// cache's tile is.
#define ATTN_Q8_TILE_WORDS (BLOOMERY_ATTN_TILE * ATTN_Q8_ROW_WORDS)
#define ATTN_Q8_TILE_SCALES (BLOOMERY_ATTN_TILE * ATTN_Q8_ROW_SCALES)
#define ATTN_Q8_PENDING_WORDS \
    ((ATTN_Q8_TILE_WORDS + BLOOMERY_ATTN_THREADS - 1u) / BLOOMERY_ATTN_THREADS)
#define ATTN_Q8_PENDING_SCALES \
    ((ATTN_Q8_TILE_SCALES + BLOOMERY_ATTN_THREADS - 1u) / BLOOMERY_ATTN_THREADS)

struct Q8Cache {
    device uint *k_codes;
    device ushort *k_scales;
    device uint *v_codes;
    device ushort *v_scales;
    threadgroup uint *tile_k_codes;
    threadgroup ushort *tile_k_scales;
    threadgroup uint *tile_v_codes;
    threadgroup ushort *tile_v_scales;

    struct Pending {
        uint k[ATTN_Q8_PENDING_WORDS];
        uint v[ATTN_Q8_PENDING_WORDS];
        ushort ks[ATTN_Q8_PENDING_SCALES];
        ushort vs[ATTN_Q8_PENDING_SCALES];
    };

    void fetch(uint tid, uint kv, uint ctx, uint lo, uint hi, thread Pending &pending) {
        for (uint i = 0u; i < ATTN_Q8_PENDING_WORDS; ++i) {
            uint e = tid + i * BLOOMERY_ATTN_THREADS;
            uint t = e / ATTN_Q8_ROW_WORDS;
            uint c = e - t * ATTN_Q8_ROW_WORDS;
            uint index = lo + t;
            uint wk = 0u;
            uint wv = 0u;
            if (e < ATTN_Q8_TILE_WORDS && index < hi) {
                uint at = (kv * ctx + index) * ATTN_Q8_ROW_WORDS + c;
                wk = k_codes[at];
                wv = v_codes[at];
            }
            pending.k[i] = wk;
            pending.v[i] = wv;
        }
        for (uint i = 0u; i < ATTN_Q8_PENDING_SCALES; ++i) {
            uint e = tid + i * BLOOMERY_ATTN_THREADS;
            uint t = e / ATTN_Q8_ROW_SCALES;
            uint b = e - t * ATTN_Q8_ROW_SCALES;
            uint index = lo + t;
            ushort ks = 0u;
            ushort vs = 0u;
            if (e < ATTN_Q8_TILE_SCALES && index < hi) {
                uint row = kv * ctx + index;
                ks = k_scales[row * ATTN_Q8_ROW_SCALES + b];
                vs = v_scales[row * ATTN_Q8_ROW_SCALES + b];
            }
            pending.ks[i] = ks;
            pending.vs[i] = vs;
        }
    }

    void commit(uint tid, thread const Pending &pending) {
        for (uint i = 0u; i < ATTN_Q8_PENDING_WORDS; ++i) {
            uint e = tid + i * BLOOMERY_ATTN_THREADS;
            if (e < ATTN_Q8_TILE_WORDS) {
                tile_k_codes[e] = pending.k[i];
                tile_v_codes[e] = pending.v[i];
            }
        }
        for (uint i = 0u; i < ATTN_Q8_PENDING_SCALES; ++i) {
            uint e = tid + i * BLOOMERY_ATTN_THREADS;
            if (e < ATTN_Q8_TILE_SCALES) {
                tile_k_scales[e] = pending.ks[i];
                tile_v_scales[e] = pending.vs[i];
            }
        }
    }

    // The lane's code words, then each word's four values in byte order.
    static void lane_values(threadgroup uint const *codes,
                            threadgroup ushort const *scales,
                            uint t,
                            uint lane,
                            thread float out[BLOOMERY_ATTN_LANE_DIMS]) {
        ushort s = scales[t * ATTN_Q8_ROW_SCALES + lane / ATTN_Q8_BLOCK_LANES];
        uint words[ATTN_Q8_LANE_WORDS];
        for (uint w = 0u; w < ATTN_Q8_LANE_WORDS; ++w) {
            words[w] = codes[t * ATTN_Q8_ROW_WORDS + lane * ATTN_Q8_LANE_WORDS + w];
        }
        for (uint w = 0u; w < ATTN_Q8_LANE_WORDS; ++w) {
            for (uint d = 0u; d < 4u; ++d) {
                out[4u * w + d] = attn_dequant(words[w], s, d);
            }
        }
    }

    void key(uint t, uint lane, thread float out[BLOOMERY_ATTN_LANE_DIMS]) {
        lane_values(tile_k_codes, tile_k_scales, t, lane, out);
    }

    void value(uint t, uint lane, thread float out[BLOOMERY_ATTN_LANE_DIMS]) {
        lane_values(tile_v_codes, tile_v_scales, t, lane, out);
    }

    void store_k(thread float const row[BLOOMERY_ATTN_LANE_DIMS], uint lane, uint cache_row,
                 device atomic_uint *faults, uint layer) {
        attn_store_q8_lanes(row, lane, k_codes, k_scales, cache_row, faults, layer);
    }

    void store_v(thread float const row[BLOOMERY_ATTN_LANE_DIMS], uint lane, uint cache_row,
                 device atomic_uint *faults, uint layer) {
        attn_store_q8_lanes(row, lane, v_codes, v_scales, cache_row, faults, layer);
    }
};

// The segment that owns the position appends that key and value. The whole
// owning SIMD group builds each row and stores it a lane at a time. The
// segments tile the keys from `win_lo`; a position before them is the first
// segment's.
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
                        uint win_lo,
                        uint sg,
                        uint lane,
                        threadgroup float *scratch) {
    if (sg != 0u) {
        return;
    }
    uint pos_seg = args.pos >= win_lo ? (args.pos - win_lo) / span : 0u;
    bool owns = ok && args.pos < args.keys && pos_seg == seg;
    if (!owns) {
        return;
    }
    uint dim0 = lane * BLOOMERY_ATTN_LANE_DIMS;
    threadgroup float *row0 = scratch;
    device float const *rope_row = rope + args.pos * BLOOMERY_ATTN_ROT;
    for (uint d = 0u; d < BLOOMERY_ATTN_LANE_DIMS; ++d) {
        row0[dim0 + d] = k[kv * BLOOMERY_ATTN_HEAD + dim0 + d];
    }
    float y[BLOOMERY_ATTN_LANE_DIMS];
    attn_norm_rope(row0, k_gain, rope_row, lane, args.eps, y);
    cache.store_k(y, lane, kv * args.ctx + args.pos, faults, args.layer);
    for (uint d = 0u; d < BLOOMERY_ATTN_LANE_DIMS; ++d) {
        y[d] = v[kv * BLOOMERY_ATTN_HEAD + dim0 + d];
    }
    cache.store_v(y, lane, kv * args.ctx + args.pos, faults, args.layer);
}

// This group's query head, normed and roped, in `qv`.
static void attn_prepare_q(thread const AttnArgs &args,
                           device float const *q,
                           device float const *q_gain,
                           device float const *rope,
                           bool ok,
                           uint kv,
                           uint sg,
                           uint lane,
                           threadgroup float *scratch,
                           thread float qv[BLOOMERY_ATTN_LANE_DIMS]) {
    for (uint d = 0u; d < BLOOMERY_ATTN_LANE_DIMS; ++d) {
        qv[d] = 0.0f;
    }
    if (!ok) {
        return;
    }
    uint head = kv * BLOOMERY_ATTN_GQA + sg * BLOOMERY_ATTN_HPT;
    uint dim0 = lane * BLOOMERY_ATTN_LANE_DIMS;
    threadgroup float *row = scratch + sg * BLOOMERY_ATTN_HEAD;
    for (uint d = 0u; d < BLOOMERY_ATTN_LANE_DIMS; ++d) {
        row[dim0 + d] = q[head * BLOOMERY_ATTN_QG + dim0 + d];
    }
    attn_norm_rope(row, q_gain, rope + args.pos * BLOOMERY_ATTN_ROT, lane, args.eps, qv);
}

// Online softmax over this segment's tiles, the keys streamed a row a
// lane. The running max, weight sum and values stay in registers. A tile's
// rows are stored to threadgroup memory after the previous tile's arithmetic
// and read by the SIMD groups of the key head, which share no state: the
// barriers are the tile's own. A key past the segment's live end (`hi`) is a
// zero row: its score is -infinity and its weight exactly 0, so the loop has
// no branch on a key and its dead fma adds an exact 0.
template <typename Cache>
static void attn_tile_loop(Cache cache,
                           thread const AttnArgs &args,
                           bool ok,
                           uint kv,
                           uint seg,
                           uint span,
                           uint win_lo,
                           uint max_live,
                           uint tid,
                           uint lane,
                           thread float qv[BLOOMERY_ATTN_LANE_DIMS],
                           thread float *running_m,
                           thread float *running_l,
                           thread float acc[BLOOMERY_ATTN_LANE_DIMS]) {
    *running_m = -INFINITY;
    *running_l = 0.0f;
    for (uint d = 0u; d < BLOOMERY_ATTN_LANE_DIMS; ++d) {
        acc[d] = 0.0f;
    }
    if (!ok) {
        return;
    }
    uint lo0 = win_lo + seg * span;
    uint seg_hi = min(win_lo + (seg + 1u) * span, max_live);
    typename Cache::Pending ahead;
    cache.fetch(tid, kv, args.ctx, lo0, min(lo0 + BLOOMERY_ATTN_TILE, seg_hi), ahead);
    for (uint lo = lo0; lo < seg_hi; lo += BLOOMERY_ATTN_TILE) {
        uint hi = min(lo + BLOOMERY_ATTN_TILE, seg_hi);
        cache.commit(tid, ahead);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint next = lo + BLOOMERY_ATTN_TILE;
        if (next < seg_hi) {
            cache.fetch(tid, kv, args.ctx, next, min(next + BLOOMERY_ATTN_TILE, seg_hi), ahead);
        }
        float scores[BLOOMERY_ATTN_TILE];
        float tile_max = -INFINITY;
        for (uint t = 0u; t < BLOOMERY_ATTN_TILE; ++t) {
            float kvv[BLOOMERY_ATTN_LANE_DIMS];
            cache.key(t, lane, kvv);
            float dot = 0.0f;
            for (uint d = 0u; d < BLOOMERY_ATTN_LANE_DIMS; ++d) {
                dot = fma(qv[d], kvv[d], dot);
            }
            dot = simd_sum(dot);
            float score = dot * args.scale;
            scores[t] = lo + t < hi ? score : -INFINITY;
            tile_max = max(tile_max, scores[t]);
        }
        float new_m = max(*running_m, tile_max);
        float old_m = *running_m;
        float factor = old_m > -INFINITY ? exp(old_m - new_m) : 0.0f;
        *running_l *= factor;
        for (uint d = 0u; d < BLOOMERY_ATTN_LANE_DIMS; ++d) {
            acc[d] *= factor;
        }
        *running_m = new_m;
        for (uint t = 0u; t < BLOOMERY_ATTN_TILE; ++t) {
            float kvv[BLOOMERY_ATTN_LANE_DIMS];
            cache.value(t, lane, kvv);
            float weight = lo + t < hi ? exp(scores[t] - new_m) : 0.0f;
            *running_l += weight;
            for (uint d = 0u; d < BLOOMERY_ATTN_LANE_DIMS; ++d) {
                acc[d] = fma(weight, kvv[d], acc[d]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// One segment writes the output, gated when the layer has a gate. More than
// one writes the partial. `gate_src` is where the gate is read from: the
// query rows when it is packed, the per-head gate's row otherwise.
static void attn_epilogue(thread const AttnArgs &args,
                          device float const *gate_src,
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
                          thread float acc[BLOOMERY_ATTN_LANE_DIMS]) {
    uint dim0 = lane * BLOOMERY_ATTN_LANE_DIMS;
    uint head = kv * BLOOMERY_ATTN_GQA + sg * BLOOMERY_ATTN_HPT;
    if (args.segments == 1u) {
        uint o = head * BLOOMERY_ATTN_HEAD + dim0;
        if (!ok || !(*running_l > 0.0f) || !isfinite(*running_m)) {
            for (uint d = 0u; d < BLOOMERY_ATTN_LANE_DIMS; ++d) {
                out[o + d] = NAN;
            }
            return;
        }
        float inv = 1.0f / *running_l;
        for (uint d = 0u; d < BLOOMERY_ATTN_LANE_DIMS; ++d) {
            float attn = acc[d] * inv;
#if ATTN_GATE
            float g = attn_gate_value(gate_src, head, dim0 + d);
            float gated = attn * attn_sigmoid(g);
            attn_commit(out, o + d, attn, g, gated, faults, args.layer);
#else
            attn_commit_plain(out, o + d, attn);
#endif
        }
        return;
    }
    uint base = (head * args.segments + seg) * BLOOMERY_ATTN_PARTIAL;
    if (!ok) {
        if (lane == 0u) {
            partials[base] = NAN;
            partials[base + 1u] = NAN;
        }
        for (uint d = 0u; d < BLOOMERY_ATTN_LANE_DIMS; ++d) {
            partials[base + 2u + dim0 + d] = NAN;
        }
        return;
    }
    if (lane == 0u) {
        partials[base] = *running_m;
        partials[base + 1u] = *running_l;
    }
    for (uint d = 0u; d < BLOOMERY_ATTN_LANE_DIMS; ++d) {
        partials[base + 2u + dim0 + d] = acc[d];
    }
}

// The merge, folded into the flash: the dispatch's last threadgroup folds
// every head's segment partials in ascending segment order, applies the
// gate when the layer has one and writes the output, then resets the
// ticket. A head is ATTN_FOLD_SUBS threads, each holding ATTN_FOLD_VALS of its
// values (the values sub, sub + SUBS, ...). The arithmetic is the merge
// dispatch's, per head, in two passes. The first stages every head's segment
// maxima and weight sums in the shared block at once, then one thread a head
// walks them and names each segment's rescale factor in place: a kind (1 the
// segment's maximum is the new one, 0 it is not, 2 it holds no weight) and the
// factor. The running max at a segment is the prefix maximum of the earlier
// segments' maxima, so no factor's exp reads another's. The second pass
// replays the merge's fma order over the values with those factors. Each
// head's final max, weight sum and refusal sit after the factors, three floats
// a head. A NaN partial (a refused position) writes NaN and stops its head.
// An empty partial is skipped.
//
// The reads are the ticket's coherent ones: the writers fence before they
// arrive, and the acquire fence plus a coherent read is the visibility the
// protocol gives (a plain read after the fence is not, on this hardware:
// it served the zero page).
static void attn_fold(device coherent(device) float *partials,
                      device atomic_uint *ticket,
                      thread const AttnArgs &args,
                      device float const *gate_src,
                      device float *out,
                      device atomic_uint *faults,
                      uint tid,
                      threadgroup float *factors) {
    ticket_acquire();
    uint head = tid / ATTN_FOLD_SUBS;
    uint sub = tid % ATTN_FOLD_SUBS;
    threadgroup float *mine = factors + head * args.segments * 2u;
    threadgroup float *header = factors + BLOOMERY_ATTN_Q * args.segments * 2u;
    for (uint p = tid; p < BLOOMERY_ATTN_Q * args.segments; p += BLOOMERY_ATTN_THREADS) {
        uint base = p * BLOOMERY_ATTN_PARTIAL;
        factors[2u * p] = ticket_load(partials, base);
        factors[2u * p + 1u] = ticket_load(partials, base + 1u);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sub == 0u) {
        float m = -INFINITY;
        float l = 0.0f;
        bool refused = false;
        for (uint seg = 0u; seg < args.segments; ++seg) {
            float mj = mine[2u * seg];
            float lj = mine[2u * seg + 1u];
            if (isnan(mj) || isnan(lj)) {
                refused = true;
                break;
            }
            if (!(lj > 0.0f)) {
                mine[2u * seg] = 2.0f;
                continue;
            }
            if (mj > m) {
                // Up: the values enter raw and the accumulator rescales. The
                // kind word is one, so a first segment's factor of zero is a
                // factor, not the other branch.
                float f = m > -INFINITY ? exp(m - mj) : 0.0f;
                mine[2u * seg] = 1.0f;
                mine[2u * seg + 1u] = f;
                m = mj;
                l = fma(l, f, lj);
            } else {
                // Down: the values rescale and the accumulator stays.
                float g = exp(mj - m);
                mine[2u * seg] = 0.0f;
                mine[2u * seg + 1u] = g;
                l = fma(lj, g, l);
            }
        }
        header[3u * head] = m;
        header[3u * head + 1u] = l;
        header[3u * head + 2u] = refused ? 1.0f : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float m = header[3u * head];
    float l = header[3u * head + 1u];
    bool refused = header[3u * head + 2u] != 0.0f;
    float acc[ATTN_FOLD_VALS];
    for (uint d = 0u; d < ATTN_FOLD_VALS; ++d) {
        acc[d] = 0.0f;
    }
    if (!refused) {
        for (uint seg = 0u; seg < args.segments; ++seg) {
            float kind = mine[2u * seg];
            if (kind == 2.0f) {
                continue;
            }
            uint base = (head * args.segments + seg) * BLOOMERY_ATTN_PARTIAL + 2u + sub;
            bool up = kind != 0.0f;
            float f = mine[2u * seg + 1u];
            for (uint d = 0u; d < ATTN_FOLD_VALS; ++d) {
                float vj = ticket_load(partials, base + d * ATTN_FOLD_SUBS);
                if (up) {
                    acc[d] = fma(acc[d], f, vj);
                } else {
                    acc[d] = fma(vj, f, acc[d]);
                }
            }
        }
    }
    uint o = head * BLOOMERY_ATTN_HEAD;
    if (refused || !(l > 0.0f) || !isfinite(m)) {
        for (uint d = 0u; d < ATTN_FOLD_VALS; ++d) {
            out[o + d * ATTN_FOLD_SUBS + sub] = NAN;
        }
    } else {
        float inv = 1.0f / l;
        for (uint d = 0u; d < ATTN_FOLD_VALS; ++d) {
            uint dim = d * ATTN_FOLD_SUBS + sub;
            float attn = acc[d] * inv;
#if ATTN_GATE
            float gate = attn_gate_value(gate_src, head, dim);
            float merged = attn * attn_sigmoid(gate);
            attn_commit(out, o + dim, attn, gate, merged, faults, args.layer);
#else
            attn_commit_plain(out, o + dim, attn);
#endif
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
// `hgate` is this position's row of the per-head gate buffer, null when the
// layer has none.
template <typename Cache, bool APPEND>
static void attn_flash(Cache cache,
                       thread const AttnArgs &args,
                       device float const *q,
                       device float const *k,
                       device float const *v,
                       device float const *q_gain,
                       device float const *k_gain,
                       device float const *rope,
                       device float const *hgate,
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
#if BLOOMERY_ATTN_HEAD_GATE
    device float const *gate_src = hgate;
#else
    device float const *gate_src = q;
#endif

    uint max_live;
    bool ok = attn_bounds(args, faults, active, tid, max_live);

    // The segments tile the keys the layer attends, the last `window` of the
    // live ones.
    uint win_lo = attn_window_lo(args, max_live);
    uint span = attn_span(max_live - win_lo, args.segments);
    if constexpr (APPEND) {
        if (active) {
            attn_append(cache, args, k, v, k_gain, rope, faults, ok, kv, seg, span, win_lo, sg,
                        lane, scratch);
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    }

    if (active) {
        float qv[BLOOMERY_ATTN_LANE_DIMS];
        attn_prepare_q(args, q, q_gain, rope, ok, kv, sg, lane, scratch, qv);
        float running_m;
        float running_l;
        float acc[BLOOMERY_ATTN_LANE_DIMS];
        // The tiles load over the scratch the prepare paired through, so
        // every thread's scratch reads end here.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        attn_tile_loop(cache, args, ok, kv, seg, span, win_lo, max_live, tid, lane, qv,
                       &running_m, &running_l, acc);
        attn_epilogue(args, gate_src, partials, out, faults, ok, kv, seg, sg, lane, &running_m,
                      &running_l, acc);
    }

    // Past one segment the last threadgroup folds the partials; the ticket
    // is the dispatch's own, and the fold resets it for the next launch.
    if (args.segments > 1u) {
        ticket_is_last(ticket, groups, tid, last_flag);
        if (last_flag == 1u) {
            attn_fold(partials, ticket, args, gate_src, out, faults, tid,
                      (threadgroup float *)scratch);
        }
    }
}

// The threadgroup's one shared block, in 16-byte words: the scratch the
// append and the q prepare pair through, then — after the barrier that ends
// the prepare — the K and V tiles the loop loads over it, then the fold's
// factors. The scratch is the first eight heads' floats, inside tile_k;
// tile_v follows tile_k, so the scratch and tile_v never share a word.
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
                             uint3 tpg [[threadgroups_per_grid]] ATTN_HGATE_PARAM(13)) {
    threadgroup uint4 shared[ATTN_SHARED_WORDS];
    threadgroup uint last_flag = 0u;
    AttnArgs args = launch;
    F16Cache cache;
    cache.k = cache_k;
    cache.v = cache_v;
    cache.tile_k = (threadgroup attn_lane_word *)shared;
    cache.tile_v = cache.tile_k + BLOOMERY_ATTN_TILE * BLOOMERY_ATTN_LANES;
    attn_flash<F16Cache, true>(cache, args, q, k, v, q_gain, k_gain, rope, ATTN_HGATE(0u),
                               partials, out, faults, ticket, tpg.x, group.x, tid,
                               (threadgroup float *)shared, last_flag);
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
                            uint3 tpg [[threadgroups_per_grid]] ATTN_HGATE_PARAM(15)) {
    // The q8_0 tiles are a row's code words and scales a tile row, K then V,
    // the scales given four 16-byte words a tile row; the shared block covers
    // the f16 tiles, which the scratch also pairs through.
    threadgroup uint4 shared[ATTN_SHARED_WORDS];
    threadgroup uint last_flag = 0u;
    AttnArgs args = launch;
    Q8Cache cache;
    cache.k_codes = k_codes;
    cache.k_scales = k_scales;
    cache.v_codes = v_codes;
    cache.v_scales = v_scales;
    constexpr uint codes_words = ATTN_Q8_ROW_WORDS / 4u;
    cache.tile_k_codes = (threadgroup uint *)shared;
    cache.tile_k_scales = (threadgroup ushort *)(shared + BLOOMERY_ATTN_TILE * codes_words);
    cache.tile_v_codes = (threadgroup uint *)(shared + BLOOMERY_ATTN_TILE * (codes_words + 4u));
    cache.tile_v_scales =
        (threadgroup ushort *)(shared + BLOOMERY_ATTN_TILE * (2u * codes_words + 4u));
    attn_flash<Q8Cache, true>(cache, args, q, k, v, q_gain, k_gain, rope, ATTN_HGATE(0u),
                              partials, out, faults, ticket, tpg.x, group.x, tid,
                              (threadgroup float *)shared, last_flag);
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
                ok, kv, 0u, span, 0u, 0u, lane, scratch);
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
                                  uint3 tpg [[threadgroups_per_grid]] ATTN_HGATE_PARAM(10)) {
    threadgroup uint4 shared[ATTN_SHARED_WORDS];
    threadgroup uint last_flag = 0u;
    F16Cache cache;
    cache.k = cache_k;
    cache.v = cache_v;
    cache.tile_k = (threadgroup attn_lane_word *)shared;
    cache.tile_v = cache.tile_k + BLOOMERY_ATTN_TILE * BLOOMERY_ATTN_LANES;
    uint row = group.y;
    AttnArgs a = attn_row_args(args, row);
    uint partials_stride = BLOOMERY_ATTN_Q * args.segments * BLOOMERY_ATTN_PARTIAL;
    attn_flash<F16Cache, false>(cache, a, q + row * args.q_stride, nullptr, nullptr, q_gain,
                                nullptr, rope, ATTN_HGATE(row), partials + row * partials_stride,
                                out + row * args.out_stride, faults, ticket + row, tpg.x, group.x,
                                tid, (threadgroup float *)shared, last_flag);
}

// The flash of a chunk of rows with simdgroup matrices. A threadgroup holds
// four consecutive positions of one key head, a SIMD group each; a SIMD
// group's rows are the query heads of the key head at its position, padded
// with zero queries to the fragment's 8 rows when the key head has fewer, so
// the 8x8 fragments of its score matrix are (heads x keys) and every key of a
// tile serves all the heads. A padded row has a zero query, so its scores,
// weights and values are finite and no other row reads them, and it stores
// nothing. The causal mask is therefore one position a SIMD
// group, never a row. The K and V tiles of 16 keys sit in threadgroup memory
// beside the other three SIMD groups' and are the loop's only shared state:
// the queries, the running max and weight sum and the head's values of each
// row stay in the SIMD group's registers.
//
// The queries are normed and roped by the decode's own functions into a
// threadgroup stage, rounded to f16, and held as fragments for the whole
// loop. The score is the fragment product of f16 operands in f32 over the
// head's dims, times the decode's scale. The softmax is online in f32 over tiles of
// 16 keys taken from absolute key 0, so a row's tiles do not depend on the
// chunk it came in or on the rows around it; the weights of a tile are
// rounded to f16 for the value product, and the weight sum is the f32
// weights'. A tile beyond a row's own position is an exact no-op for the
// row: its weights are 0 and its rescale factor 1. The sums are therefore not
// the decode flash's: the rows are held to the decode flash by a band, not by
// bits.
//
// A query value past f16's range raises the product site and the SIMD
// group's rows are NaN. A key or value row that is not finite makes NaN the
// rows of its tile and after, and raises nothing, as in the decode flash. The
// tail of the chunk repeats its last row in the SIMD groups past it, and
// stores nothing for them.
#include <metal_simdgroup_matrix>

#define PF_STRIDE (BLOOMERY_ATTN_HEAD + BLOOMERY_ATTN_PF_PAD)
#define PF_TILE_HALVES (BLOOMERY_ATTN_PF_KEYS * PF_STRIDE)
// The rows of a fragment: the query heads of a key head, padded to eight.
#define PF_ROWS 8u
#define PF_STAGE_ROWS (BLOOMERY_ATTN_PF_POSITIONS * PF_ROWS)
// Every loop over the fragments of the query or the accumulator is unrolled
// in full: a fragment array indexed by a run-time value lives in memory, and
// the products then run at a fraction of the matrix unit's rate.
#define PF_DCHUNKS (BLOOMERY_ATTN_HEAD / 8u)
// A cache row in 16-byte words, and the loads a thread makes to fill a tile.
#define PF_ROW_WORDS (BLOOMERY_ATTN_HEAD / 8u)
#define PF_LOADS (BLOOMERY_ATTN_PF_KEYS * PF_ROW_WORDS / (BLOOMERY_ATTN_PF_POSITIONS * 32u))

static_assert(PF_STAGE_ROWS * PF_STRIDE <= 2u * PF_TILE_HALVES,
              "the query stage fits in the K and V tiles it shares memory with");
static_assert(BLOOMERY_ATTN_PF_KEYS == 16u, "a tile is two 8-key fragments");
static_assert(BLOOMERY_ATTN_GQA <= PF_ROWS, "a fragment's rows hold a key head's query heads");
static_assert(PF_LOADS * BLOOMERY_ATTN_PF_POSITIONS * 32u == BLOOMERY_ATTN_PF_KEYS * PF_ROW_WORDS,
              "the threadgroup's loads cover a tile exactly");

kernel void attn_prefill_f16(device float const *q [[buffer(0)]],
                             device float const *q_gain [[buffer(1)]],
                             device float const *rope [[buffer(2)]],
                             device ushort const *cache_k [[buffer(3)]],
                             device ushort const *cache_v [[buffer(4)]],
                             device float *out [[buffer(5)]],
                             constant AttnRowsArgs &args [[buffer(6)]],
                             device atomic_uint *faults [[buffer(7)]],
                             uint tid [[thread_index_in_threadgroup]],
                             uint sg [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint3 group [[threadgroup_position_in_grid]],
                             uint3 tpg [[threadgroups_per_grid]] ATTN_HGATE_PARAM(8)) {
    threadgroup half shared[2u * PF_TILE_HALVES];
    threadgroup float scratch[BLOOMERY_ATTN_PF_POSITIONS * BLOOMERY_ATTN_HEAD];
    threadgroup half *tile_k = shared;
    threadgroup half *tile_v = shared + PF_TILE_HALVES;

    // The heaviest blocks (the last positions) are the first threadgroups.
    uint kv = group.y;
    uint row0 = (tpg.x - 1u - group.x) * BLOOMERY_ATTN_PF_POSITIONS;
    uint last = min(row0 + BLOOMERY_ATTN_PF_POSITIONS, args.rows) - 1u;
    bool writes = row0 + sg <= last;
    uint row = min(row0 + sg, last);
    AttnArgs a = attn_row_args(args, row);
    uint max_live;
    bool ok = attn_bounds(a, faults, writes, lane, max_live);
    uint block_keys = args.from + last + 1u;
    // A block past the cache has no key to read; its rows are the bounds' NaN.
    if (block_keys > args.ctx) {
        block_keys = 0u;
    }
#if BLOOMERY_ATTN_WINDOW
    // The first key this row attends, and the first tile of the block: the one
    // holding the window of its first row, the earliest of the block's.
    AttnArgs first = attn_row_args(args, row0);
    uint row_lo = attn_window_lo(a, a.keys);
    uint first_tile = attn_window_lo(first, first.keys) / BLOOMERY_ATTN_PF_KEYS;
#else
    uint first_tile = 0u;
#endif

    // The queries of this position, normed and roped, as f16 rows of the stage.
    threadgroup float *sc = scratch + sg * BLOOMERY_ATTN_HEAD;
    bool overflow = false;
    bool unusable = !ok;
    for (uint h = 0u; h < PF_ROWS; ++h) {
        uint head = kv * BLOOMERY_ATTN_GQA + h;
        float y[BLOOMERY_ATTN_LANE_DIMS];
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (ok && h < BLOOMERY_ATTN_GQA) {
            device float const *src = q + (ulong)row * args.q_stride + head * BLOOMERY_ATTN_QG +
                                      lane * BLOOMERY_ATTN_LANE_DIMS;
            for (uint d = 0u; d < BLOOMERY_ATTN_LANE_DIMS; ++d) {
                sc[lane * BLOOMERY_ATTN_LANE_DIMS + d] = src[d];
            }
            attn_norm_rope(sc, q_gain, rope + a.pos * BLOOMERY_ATTN_ROT, lane, a.eps, y);
        } else {
            for (uint d = 0u; d < BLOOMERY_ATTN_LANE_DIMS; ++d) {
                y[d] = 0.0f;
            }
        }
        threadgroup half *dst = shared + (sg * PF_ROWS + h) * PF_STRIDE +
                                lane * BLOOMERY_ATTN_LANE_DIMS;
        for (uint d = 0u; d < BLOOMERY_ATTN_LANE_DIMS; ++d) {
            half hv = half(y[d]);
            if (!isfinite(float(hv))) {
                unusable = true;
                overflow = overflow || isfinite(y[d]);
            }
            dst[d] = hv;
        }
    }
    unusable = simd_any(unusable);
    overflow = simd_any(overflow);
    if (overflow && lane == 0u && writes) {
        raise(faults, a.layer, BLOOMERY_FAULT_SITE_F32_PRODUCT);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_half8x8 qf[PF_DCHUNKS];
    #pragma clang loop unroll(full)
    for (uint c = 0u; c < PF_DCHUNKS; ++c) {
        simdgroup_load(qf[c], shared + sg * PF_ROWS * PF_STRIDE + c * 8u, PF_STRIDE);
    }
    // The K and V tiles are loaded over the stage.
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Where this lane sits in a fragment: row `fm` (a query head), columns
    // `fn` and `fn + 1`.
    ushort qid = ushort(lane / 4u);
    ushort fm = (qid & 4u) + ((lane / 2u) % 4u);
    ushort fn = (qid & 2u) * 2u + (lane % 2u) * 2u;

    simdgroup_float8x8 acc[PF_DCHUNKS];
    #pragma clang loop unroll(full)
    for (uint c = 0u; c < PF_DCHUNKS; ++c) {
        acc[c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    float running_m = -INFINITY;
    float running_l = 0.0f;

    uint tiles = (block_keys + BLOOMERY_ATTN_PF_KEYS - 1u) / BLOOMERY_ATTN_PF_KEYS;
    for (uint t = first_tile; t < tiles; ++t) {
        uint k0 = t * BLOOMERY_ATTN_PF_KEYS;
        for (uint i = 0u; i < PF_LOADS; ++i) {
            uint e = tid + i * (BLOOMERY_ATTN_PF_POSITIONS * 32u);
            uint kr = e / PF_ROW_WORDS;
            uint c = e % PF_ROW_WORDS;
            uint key = k0 + kr;
            uint4 wk = uint4(0u);
            uint4 wv = uint4(0u);
            if (key < block_keys) {
                ulong at = ((ulong)kv * args.ctx + key) * PF_ROW_WORDS + c;
                wk = ((device const uint4 *)cache_k)[at];
                wv = ((device const uint4 *)cache_v)[at];
            }
            *((threadgroup uint4 *)(tile_k + kr * PF_STRIDE + c * 8u)) = wk;
            *((threadgroup uint4 *)(tile_v + kr * PF_STRIDE + c * 8u)) = wv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 s0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_float8x8 s1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        #pragma clang loop unroll(full)
    for (uint c = 0u; c < PF_DCHUNKS; ++c) {
            simdgroup_half8x8 k0f;
            simdgroup_half8x8 k1f;
            simdgroup_load(k0f, tile_k + c * 8u, PF_STRIDE, ulong2(0, 0), true);
            simdgroup_load(k1f, tile_k + 8u * PF_STRIDE + c * 8u, PF_STRIDE, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(s0, qf[c], k0f, s0);
            simdgroup_multiply_accumulate(s1, qf[c], k1f, s1);
        }

        // This lane's four scores of its row: keys fn, fn + 1 of each fragment.
        float sv[4];
        bool live[4];
        float tile_max = -INFINITY;
        for (uint j = 0u; j < 4u; ++j) {
            float raw = j < 2u ? s0.thread_elements()[j] : s1.thread_elements()[j - 2u];
            uint key = k0 + (j / 2u) * 8u + fn + (j % 2u);
            live[j] = key <= a.pos;
#if BLOOMERY_ATTN_WINDOW
            live[j] = live[j] && key >= row_lo;
#endif
            sv[j] = raw * a.scale;
            if (live[j]) {
                tile_max = max(tile_max, sv[j]);
            }
        }
        tile_max = max(tile_max, simd_shuffle_xor(tile_max, ushort(1)));
        tile_max = max(tile_max, simd_shuffle_xor(tile_max, ushort(8)));
        float new_m = max(running_m, tile_max);
        float old_m = running_m;
        float factor = old_m > -INFINITY ? exp(old_m - new_m) : 0.0f;
        float p[4];
        float weights = 0.0f;
        for (uint j = 0u; j < 4u; ++j) {
            p[j] = live[j] ? exp(sv[j] - new_m) : 0.0f;
            weights += p[j];
        }
        running_l = running_l * factor + weights;
        running_m = new_m;
        // The rows' rescale as a diagonal fragment product, so the accumulators
        // stay in matrix registers; a tile that raised no row's max skips it.
        if (t > 0u && simd_any(factor != 1.0f)) {
            simdgroup_float8x8 diag;
            diag.thread_elements()[0] = fn == fm ? factor : 0.0f;
            diag.thread_elements()[1] = fn + 1u == fm ? factor : 0.0f;
            #pragma clang loop unroll(full)
    for (uint c = 0u; c < PF_DCHUNKS; ++c) {
                simdgroup_float8x8 scaled;
                simdgroup_multiply(scaled, diag, acc[c]);
                acc[c] = scaled;
            }
        }
        simdgroup_half8x8 p0;
        simdgroup_half8x8 p1;
        p0.thread_elements()[0] = half(p[0]);
        p0.thread_elements()[1] = half(p[1]);
        p1.thread_elements()[0] = half(p[2]);
        p1.thread_elements()[1] = half(p[3]);
        #pragma clang loop unroll(full)
    for (uint c = 0u; c < PF_DCHUNKS; ++c) {
            simdgroup_half8x8 v0f;
            simdgroup_half8x8 v1f;
            simdgroup_load(v0f, tile_v + c * 8u, PF_STRIDE);
            simdgroup_load(v1f, tile_v + 8u * PF_STRIDE + c * 8u, PF_STRIDE);
            simdgroup_multiply_accumulate(acc[c], p0, v0f, acc[c]);
            simdgroup_multiply_accumulate(acc[c], p1, v1f, acc[c]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // The weight sum of a row is the sum of its four lanes' partial sums.
    running_l += simd_shuffle_xor(running_l, ushort(1));
    running_l += simd_shuffle_xor(running_l, ushort(8));
    if (!writes || fm >= BLOOMERY_ATTN_GQA) {
        return;
    }
    uint head = kv * BLOOMERY_ATTN_GQA + fm;
    bool whole = !unusable && running_l > 0.0f && isfinite(running_m);
    float inv = 1.0f / running_l;
#if ATTN_GATE
#if BLOOMERY_ATTN_HEAD_GATE
    device float const *gate_row = hgate + (ulong)row * BLOOMERY_ATTN_Q;
#else
    device float const *gate_row = q + (ulong)row * args.q_stride;
#endif
#endif
    #pragma clang loop unroll(full)
    for (uint c = 0u; c < PF_DCHUNKS; ++c) {
        for (uint e = 0u; e < 2u; ++e) {
            uint dim = c * 8u + fn + e;
            uint at = row * args.out_stride + head * BLOOMERY_ATTN_HEAD + dim;
            if (!whole) {
                out[at] = NAN;
                continue;
            }
            float attn = acc[c].thread_elements()[e] * inv;
#if ATTN_GATE
            float g = attn_gate_value(gate_row, head, dim);
            float product = attn * attn_sigmoid(g);
            attn_commit(out, at, attn, g, product, faults, a.layer);
#else
            attn_commit_plain(out, at, attn);
#endif
        }
    }
}
