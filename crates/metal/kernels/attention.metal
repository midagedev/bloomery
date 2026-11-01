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
// partial per segment, and the last threadgroup of each key head — its ticket
// word's — folds that key head's partials, applies the gate and writes its
// query heads' output, in the merge's arithmetic, then resets the word. The
// key heads fold at once, one query head a SIMD group. There is no merge
// dispatch. The f16 cache's one segment is the one-pass flash instead
// (attn_flash_one_f16, after the rows launches): a threadgroup a query head
// over every key, described there.
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
// The prologue has two compile-time axes (BLOOMERY_ATTN_QK_NORM,
// BLOOMERY_ATTN_PAIRS_NORM). With no head norm no norm math runs at all: no
// sum of squares, no gain. With NORM pairs the rope turns the neighbours
// `2i, 2i + 1` instead of the NEOX halves `i, i + ROT/2`, by the same angle
// table; a pair never leaves a lane (a lane's dims are even), so the lane
// reads only its own writes, no barrier orders the prologue and the pair
// loads as one `float2`. The head norm's sum-of-squares walks the NEOX pair
// tree, so a head norm with NORM pairs and a norm-free NEOX prologue are no
// entry of this file.
//
// The score is each lane's fmas in dim order, then 31 adds across the lanes
// (the segmented flash's xor butterfly, the one-pass flash's SIMD-group sum,
// whichever tree the language picks for it), then the scale (exact at head
// 256, ik's f32 1/sqrt(head) otherwise), as the band in the oracle derives.
// The softmax rescales once a 16-key tile; the fold adds the segments'
// partials in ascending segment order. The value sum is one fma a key in key
// order, per dim.
//
// A tile holds no branch on a key's liveness: a key past the segment's end
// is a zero row, scores -infinity and weighs exactly 0, so the tile's sixteen
// keys are one straight line the compiler schedules together. Each lane reads
// its own word of the tile's rows from the cache into registers, four rows in
// flight, and keeps one key's score: a tile's weights are one exp a lane.
//
// The rows launches run a chunk of causal queries. attn_norm_rope_rows appends
// row r's key and value at position from + r, by the same append the decode
// flash runs, one SIMD group a (key head, row). gqa256_flash_f16_rows is the
// decode flash with a row grid dimension: row r is the launch at position
// from + r over from + r + 1 keys, with its own partials and ticket words, and
// it reads the rows the append dispatch wrote. A row is the bits of the decode
// launch at its position with the same segment count; at one segment both are
// the one-pass flash's.
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
static_assert(BLOOMERY_ATTN_QK_NORM == 0u || BLOOMERY_ATTN_QK_NORM == 1u,
              "a layer's query and key heads are normed by their gains or they are not");
static_assert(BLOOMERY_ATTN_PAIRS_NORM == 0u || BLOOMERY_ATTN_PAIRS_NORM == 1u,
              "the rope pairs NEOX halves or it pairs neighbours");
static_assert(!(BLOOMERY_ATTN_QK_NORM && BLOOMERY_ATTN_PAIRS_NORM),
              "the head norm's sum-of-squares walks the NEOX pair tree; a head norm with "
              "neighbour pairs is no entry of this file");
#if !BLOOMERY_ATTN_PAIRS_NORM
static_assert(BLOOMERY_ATTN_HEAD % 64u == 0u &&
                  (BLOOMERY_ATTN_ROT / 2u <= BLOOMERY_ATTN_LANES ||
                   BLOOMERY_ATTN_ROT == BLOOMERY_ATTN_HEAD),
              "the norm's pair tree: whole warps, the turning pairs in warp 0 or every value");
#endif

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

// The fold's segment slots: lane s of a folding SIMD group holds segments s,
// s + 32, ... of its head.
#define ATTN_FOLD_ROUNDS ((BLOOMERY_ATTN_SEGMENT_CAP + 31u) / 32u)

// The flash's one shared block, in 16-byte words: a head row a SIMD group,
// the scratch the append and the q prepare pair through. The walk reads the
// cache into registers and the fold holds its factors in lanes, so neither
// takes threadgroup memory.
#define ATTN_SHARED_WORDS (BLOOMERY_ATTN_SG * BLOOMERY_ATTN_HEAD / 4u)

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
// the first of them when the launch names no window (0). Only an instance whose
// flash cuts a window has it; every other instance's flash is the plain one.
#if BLOOMERY_ATTN_WINDOW
inline uint attn_window_lo(thread const AttnArgs &args, uint live) {
    return (args.window != 0u && live > args.window) ? live - args.window : 0u;
}
#endif

// The layer's output gate, packed in its query or one value a head.
#define ATTN_GATE (BLOOMERY_ATTN_GATED || BLOOMERY_ATTN_HEAD_GATE)

// The per-head gate buffer, for an instance that gates per head and for no
// other: the kernel's parameter (bound by a launch whose layer gates per head),
// the flash's parameter and argument (the row `row`'s start, before the
// partials) and the source the epilogue and the fold read the gate from, which
// is the query rows when the gate is packed in them. An instance with no
// per-head gate expands every one of them to nothing or to `q`: its kernels
// are the text they were before the axis.
#if BLOOMERY_ATTN_HEAD_GATE
#define ATTN_HGATE_PARAM(n) , device float const *hgate [[buffer(n)]]
#define ATTN_HGATE_FORMAL device float const *hgate,
#define ATTN_HGATE_ACTUAL(row) (hgate + (ulong)(row) * BLOOMERY_ATTN_Q),
#define ATTN_GATE_SRC hgate
#else
#define ATTN_HGATE_PARAM(n)
#define ATTN_HGATE_FORMAL
#define ATTN_HGATE_ACTUAL(row)
#define ATTN_GATE_SRC q
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

#if BLOOMERY_ATTN_QK_NORM
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
#endif

// Norm then rope of the head in `scratch`, into `y` (this lane's dims). The
// gain is HEAD values and the rope row ROT. `lane` owns dims
// lane*LANE_DIMS .. +LANE_DIMS-1. The two axes of the file's header pick
// the variant: the norm and the NEOX pairs, or the raw values and the
// neighbour pairs.
#if BLOOMERY_ATTN_QK_NORM
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
#elif BLOOMERY_ATTN_PAIRS_NORM
// The raw values roped as neighbours: no sum of squares, no gain read and
// no barrier, the pair `2i, 2i + 1` of the lane's own dims turned by the
// table row's pair `i` (its values sit at `2i, 2i + 1` of the row, and `i`
// is even here, so the row loads beside its pair). `gain` and `eps` are the
// norm variant's and go unread.
static void attn_norm_rope(threadgroup float *scratch,
                           device float const *gain,
                           device float const *rope_row,
                           uint lane,
                           float eps,
                           thread float y[BLOOMERY_ATTN_LANE_DIMS]) {
    uint dim0 = lane * BLOOMERY_ATTN_LANE_DIMS;
    for (uint d = 0u; d < BLOOMERY_ATTN_LANE_DIMS; d += 2u) {
        uint i = dim0 + d;
        if (i < BLOOMERY_ATTN_ROT) {
            float2 n = *(threadgroup float2 const *)(scratch + i);
            float c = rope_row[i];
            float s = rope_row[i + 1u];
            y[d] = fma(n.x, c, -(n.y * s));
            y[d + 1u] = fma(n.x, s, n.y * c);
        } else {
            y[d] = scratch[i];
            y[d + 1u] = scratch[i + 1u];
        }
    }
}
#else
#error "an instance with no head norm and NEOX pairs is no entry of this file"
#endif

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

// The f16 cache, read a row a lane: the lane's values of a row are one word
// of halves, loaded from the cache into registers. A row past the live keys
// is the zero word, the value 0.0f.
struct F16Cache {
    device ushort *k;
    device ushort *v;

    typedef attn_lane_word Word;

    static Word zero() {
        return Word(0u);
    }

    Word key_word(uint row, uint lane) {
        return ((const device attn_lane_word *)(k + row * BLOOMERY_ATTN_HEAD))[lane];
    }

    Word value_word(uint row, uint lane) {
        return ((const device attn_lane_word *)(v + row * BLOOMERY_ATTN_HEAD))[lane];
    }

    static void widen(Word w, thread float out[BLOOMERY_ATTN_LANE_DIMS]) {
        attn_widen(w, out);
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

// The q8_0 cache, read a row a lane: HEAD/4 code words and HEAD/32 scales a
// row, low byte first. A lane's values sit in its code words of one 32-value
// block, under that block's scale. A row past the live keys is zero codes and
// a zero scale, the value 0.0f.
struct Q8Word {
    uint codes[ATTN_Q8_LANE_WORDS];
    ushort scale;
};

struct Q8Cache {
    device uint *k_codes;
    device ushort *k_scales;
    device uint *v_codes;
    device ushort *v_scales;

    typedef Q8Word Word;

    static Word zero() {
        Word w;
        for (uint i = 0u; i < ATTN_Q8_LANE_WORDS; ++i) {
            w.codes[i] = 0u;
        }
        w.scale = 0u;
        return w;
    }

    static Word row_word(device uint const *codes,
                         device ushort const *scales,
                         uint row,
                         uint lane) {
        Word w;
        for (uint i = 0u; i < ATTN_Q8_LANE_WORDS; ++i) {
            w.codes[i] = codes[row * ATTN_Q8_ROW_WORDS + lane * ATTN_Q8_LANE_WORDS + i];
        }
        w.scale = scales[row * ATTN_Q8_ROW_SCALES + lane / ATTN_Q8_BLOCK_LANES];
        return w;
    }

    Word key_word(uint row, uint lane) {
        return row_word(k_codes, k_scales, row, lane);
    }

    Word value_word(uint row, uint lane) {
        return row_word(v_codes, v_scales, row, lane);
    }

    // The lane's code words, then each word's four values in byte order.
    static void widen(Word w, thread float out[BLOOMERY_ATTN_LANE_DIMS]) {
        for (uint i = 0u; i < ATTN_Q8_LANE_WORDS; ++i) {
            for (uint d = 0u; d < 4u; ++d) {
                out[4u * i + d] = attn_dequant(w.codes[i], w.scale, d);
            }
        }
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
// owning SIMD group builds each row and stores it a lane at a time. A flash
// that cuts a window tiles its segments over the keys of the window, so a
// position before them is the first segment's.
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
#if BLOOMERY_ATTN_WINDOW
    uint win_lo = attn_window_lo(args, args.keys);
    uint pos_seg = args.pos >= win_lo ? (args.pos - win_lo) / span : 0u;
    bool owns = ok && args.pos < args.keys && pos_seg == seg;
#else
    bool owns = ok && args.pos < args.keys && args.pos / span == seg;
#endif
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

// Online softmax over this segment's keys, a tile of BLOOMERY_ATTN_TILE keys
// at a time, each lane reading its word of every key and value row straight
// from the cache, ATTN_WALK_AHEAD rows in flight. Lane t keeps the score of
// the tile's key t & 15, so the tile's weights are one exp a lane and its
// rescale one exp; each weight reaches the value fmas by shuffle. The running
// max, weight sum and values stay in registers. A key past the segment's live
// end (`hi`) is a zero row: its score is -infinity and its weight exactly 0,
// so the loop has no branch on a key and its dead fma adds an exact 0. A
// whole tile (FULL) reads its rows with no test on a key.
//
// The scores sum the lanes' partial dots by a halving butterfly, one shuffle
// a key: the four rows in flight pair by lane bit 0 (xor 1), then bit 1
// (xor 2), so lane l holds key 4g + (l & 3) of row group g summed over its
// lane quad; the tile's four groups then pair by bit 2 (xor 4) and bit 3
// (xor 8), and the two halves add across xor 16. Every key's sum is the same
// xor tree over the 32 lanes, and lanes t and t + 16 end with key t's score;
// a lane past the tile's sixteen weighs nothing, since lo + lane >= hi.
#define ATTN_WALK_AHEAD 4u

static_assert(BLOOMERY_ATTN_TILE <= BLOOMERY_ATTN_LANES, "a lane keeps one tile key's score");
static_assert(BLOOMERY_ATTN_TILE % ATTN_WALK_AHEAD == 0u, "the rows in flight divide a tile");
static_assert(BLOOMERY_ATTN_TILE == 16u && ATTN_WALK_AHEAD == 4u && BLOOMERY_ATTN_LANES == 32u,
              "the score butterfly halves four rows, then four groups, over 32 lanes");

template <typename Cache, bool FULL>
static void attn_tile(Cache cache,
                      thread const AttnArgs &args,
                      uint row0,
                      uint lo,
                      uint hi,
                      uint lane,
                      thread float qv[BLOOMERY_ATTN_LANE_DIMS],
                      thread float *running_m,
                      thread float *running_l,
                      thread float acc[BLOOMERY_ATTN_LANE_DIMS]) {
    bool b0 = (lane & 1u) != 0u;
    bool b1 = (lane & 2u) != 0u;
    float quads[BLOOMERY_ATTN_TILE / ATTN_WALK_AHEAD];
    for (uint t0 = 0u; t0 < BLOOMERY_ATTN_TILE; t0 += ATTN_WALK_AHEAD) {
        typename Cache::Word words[ATTN_WALK_AHEAD];
        for (uint c = 0u; c < ATTN_WALK_AHEAD; ++c) {
            uint key = lo + t0 + c;
            words[c] = FULL || key < hi ? cache.key_word(row0 + key, lane) : Cache::zero();
        }
        float part[ATTN_WALK_AHEAD];
        for (uint c = 0u; c < ATTN_WALK_AHEAD; ++c) {
            float kvv[BLOOMERY_ATTN_LANE_DIMS];
            Cache::widen(words[c], kvv);
            float dot = 0.0f;
            for (uint d = 0u; d < BLOOMERY_ATTN_LANE_DIMS; ++d) {
                dot = fma(qv[d], kvv[d], dot);
            }
            part[c] = dot;
        }
        float pair0 = (b0 ? part[1] : part[0]) + simd_shuffle_xor(b0 ? part[0] : part[1], ushort(1));
        float pair1 = (b0 ? part[3] : part[2]) + simd_shuffle_xor(b0 ? part[2] : part[3], ushort(1));
        quads[t0 / ATTN_WALK_AHEAD] = (b1 ? pair1 : pair0) + simd_shuffle_xor(b1 ? pair0 : pair1, ushort(2));
    }
    bool b2 = (lane & 4u) != 0u;
    bool b3 = (lane & 8u) != 0u;
    float half0 = (b2 ? quads[1] : quads[0]) + simd_shuffle_xor(b2 ? quads[0] : quads[1], ushort(4));
    float half1 = (b2 ? quads[3] : quads[2]) + simd_shuffle_xor(b2 ? quads[2] : quads[3], ushort(4));
    float sixteen = (b3 ? half1 : half0) + simd_shuffle_xor(b3 ? half0 : half1, ushort(8));
    float score = (sixteen + simd_shuffle_xor(sixteen, ushort(16))) * args.scale;
    uint t = lane & (BLOOMERY_ATTN_TILE - 1u);
    float mine = FULL || lo + t < hi ? score : -INFINITY;
    float new_m = max(*running_m, simd_max(mine));
    float old_m = *running_m;
    float factor = old_m > -INFINITY ? exp(old_m - new_m) : 0.0f;
    float weight = lo + lane < hi ? exp(mine - new_m) : 0.0f;
    *running_l = *running_l * factor + simd_sum(weight);
    for (uint d = 0u; d < BLOOMERY_ATTN_LANE_DIMS; ++d) {
        acc[d] *= factor;
    }
    *running_m = new_m;
    for (uint t0 = 0u; t0 < BLOOMERY_ATTN_TILE; t0 += ATTN_WALK_AHEAD) {
        typename Cache::Word words[ATTN_WALK_AHEAD];
        for (uint c = 0u; c < ATTN_WALK_AHEAD; ++c) {
            uint key = lo + t0 + c;
            words[c] = FULL || key < hi ? cache.value_word(row0 + key, lane) : Cache::zero();
        }
        for (uint c = 0u; c < ATTN_WALK_AHEAD; ++c) {
            float w = simd_shuffle(weight, ushort(t0 + c));
            float kvv[BLOOMERY_ATTN_LANE_DIMS];
            Cache::widen(words[c], kvv);
            for (uint d = 0u; d < BLOOMERY_ATTN_LANE_DIMS; ++d) {
                acc[d] = fma(w, kvv[d], acc[d]);
            }
        }
    }
}

template <typename Cache>
static void attn_tile_loop(Cache cache,
                           thread const AttnArgs &args,
                           bool ok,
                           uint kv,
                           uint seg,
                           uint span,
                           uint max_live,
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
#if BLOOMERY_ATTN_WINDOW
    uint win_lo = attn_window_lo(args, max_live);
    uint lo0 = win_lo + seg * span;
    uint seg_hi = min(win_lo + (seg + 1u) * span, max_live);
#else
    uint lo0 = seg * span;
    uint seg_hi = min((seg + 1u) * span, max_live);
#endif
    uint row0 = kv * args.ctx;
    for (uint lo = lo0; lo < seg_hi; lo += BLOOMERY_ATTN_TILE) {
        uint hi = min(lo + BLOOMERY_ATTN_TILE, seg_hi);
        if (hi - lo == BLOOMERY_ATTN_TILE) {
            attn_tile<Cache, true>(cache, args, row0, lo, hi, lane, qv, running_m, running_l, acc);
        } else {
            attn_tile<Cache, false>(cache, args, row0, lo, hi, lane, qv, running_m, running_l, acc);
        }
    }
}

// One segment writes the output, gated when the layer has a gate. More than
// one writes the partial. `q` is where the gate is read from: the query rows
// when it is packed, the per-head gate's row otherwise (`ATTN_GATE_SRC`).
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
#if BLOOMERY_ATTN_GATED
            float g = q[head * BLOOMERY_ATTN_QG + BLOOMERY_ATTN_HEAD + dim0 + d];
#elif BLOOMERY_ATTN_HEAD_GATE
            float g = q[head];
#endif
#if ATTN_GATE
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

// The merge, folded into the flash: the last threadgroup of key head `kv`
// folds the segment partials of that key head's query heads, applies the gate
// when the layer has one and writes their output, then resets the key head's
// ticket word. SIMD group g folds the key head's query heads g, g + SG, ...
// Lane s of the group holds slots s, s + 32, ... of the head's
// segments: the head's maximum M is the largest maximum of a segment that
// holds weight, a segment's factor is exp(m - M) (0 for one that holds none)
// and the weight sum is the group's sum of each weight sum times its factor.
// A lane keeps its LANE_DIMS values of the head, each segment's values added
// by fma in ascending segment order times the factor its lane passes by
// shuffle; a segment with no weight is skipped. A segment that holds weight
// but whose factor underflows to 0 is still added: 0 times a finite value
// adds an exact 0, and 0 times a NaN value is the NaN its head must carry.
// A NaN partial (a refused position) writes NaN for its head.
//
// The reads are the ticket's coherent ones: the writers fence before they
// arrive, and the acquire fence plus a coherent read is the visibility the
// protocol gives (a plain read after the fence is not, on this hardware:
// it served the zero page).
static_assert(BLOOMERY_ATTN_SEGMENT_CAP <= ATTN_FOLD_ROUNDS * 32u,
              "a folding lane's slots cover the segment cap");

static void attn_fold(device coherent(device) float *partials,
                      device atomic_uint *ticket,
                      thread const AttnArgs &args,
                      device float const *q,
                      device float *out,
                      device atomic_uint *faults,
                      uint tid,
                      uint kv) {
    ticket_acquire();
    uint sg = tid / BLOOMERY_ATTN_LANES;
    uint lane = tid % BLOOMERY_ATTN_LANES;
    uint fold_dim0 = lane * BLOOMERY_ATTN_LANE_DIMS;
    for (uint h = sg; h < BLOOMERY_ATTN_GQA; h += BLOOMERY_ATTN_SG) {
        uint head = kv * BLOOMERY_ATTN_GQA + h;
        float ms[ATTN_FOLD_ROUNDS];
        float ls[ATTN_FOLD_ROUNDS];
        float big = -INFINITY;
        bool refused = false;
        for (uint r = 0u; r < ATTN_FOLD_ROUNDS; ++r) {
            uint slot = lane + 32u * r;
            float mj = -INFINITY;
            float lj = 0.0f;
            if (slot < args.segments) {
                uint base = (head * args.segments + slot) * BLOOMERY_ATTN_PARTIAL;
                mj = ticket_load(partials, base);
                lj = ticket_load(partials, base + 1u);
            }
            if (isnan(mj) || isnan(lj)) {
                refused = true;
            } else if (lj > 0.0f) {
                big = max(big, mj);
            }
            ms[r] = mj;
            ls[r] = lj;
        }
        refused = simd_any(refused);
        big = simd_max(big);
        float factors[ATTN_FOLD_ROUNDS];
        float weights = 0.0f;
        for (uint r = 0u; r < ATTN_FOLD_ROUNDS; ++r) {
            factors[r] = !refused && ls[r] > 0.0f ? exp(ms[r] - big) : 0.0f;
            weights = fma(ls[r], factors[r], weights);
        }
        float l = simd_sum(weights);
        float acc[BLOOMERY_ATTN_LANE_DIMS];
        for (uint d = 0u; d < BLOOMERY_ATTN_LANE_DIMS; ++d) {
            acc[d] = 0.0f;
        }
        for (uint r = 0u; r < ATTN_FOLD_ROUNDS; ++r) {
            for (uint c = 0u; c < 32u && 32u * r + c < args.segments; ++c) {
                float f = simd_shuffle(factors[r], ushort(c));
                if (simd_shuffle(ls[r], ushort(c)) == 0.0f) {
                    continue;
                }
                uint at = (head * args.segments + 32u * r + c) * BLOOMERY_ATTN_PARTIAL + 2u;
                for (uint d = 0u; d < BLOOMERY_ATTN_LANE_DIMS; ++d) {
                    float vj = ticket_load(partials, at + fold_dim0 + d);
                    acc[d] = fma(vj, f, acc[d]);
                }
            }
        }
        uint o = head * BLOOMERY_ATTN_HEAD;
        if (refused || !(l > 0.0f) || !isfinite(big)) {
            for (uint d = 0u; d < BLOOMERY_ATTN_LANE_DIMS; ++d) {
                out[o + fold_dim0 + d] = NAN;
            }
            continue;
        }
        float inv = 1.0f / l;
        for (uint d = 0u; d < BLOOMERY_ATTN_LANE_DIMS; ++d) {
            uint dim = fold_dim0 + d;
            float attn = acc[d] * inv;
#if BLOOMERY_ATTN_GATED
            float gate = q[head * BLOOMERY_ATTN_QG + BLOOMERY_ATTN_HEAD + dim];
#elif BLOOMERY_ATTN_HEAD_GATE
            float gate = q[head];
#endif
#if ATTN_GATE
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
// `hgate` (an instance that gates per head) is this position's row of the
// per-head gate buffer.
template <typename Cache, bool APPEND>
static void attn_flash(Cache cache,
                       thread const AttnArgs &args,
                       device float const *q,
                       device float const *k,
                       device float const *v,
                       device float const *q_gain,
                       device float const *k_gain,
                       device float const *rope,
                       ATTN_HGATE_FORMAL
                       device coherent(device) float *partials,
                       device float *out,
                       device atomic_uint *faults,
                       device atomic_uint *ticket,
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

#if BLOOMERY_ATTN_WINDOW
    // The segments tile the keys the layer attends, the last `window` of the
    // live ones.
    uint span = attn_span(max_live - attn_window_lo(args, max_live), args.segments);
#else
    uint span = attn_span(max_live, args.segments);
#endif
    if constexpr (APPEND) {
        if (active) {
            attn_append(cache, args, k, v, k_gain, rope, faults, ok, kv, seg, span, sg, lane,
                        scratch);
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    }

    if (active) {
        float qv[BLOOMERY_ATTN_LANE_DIMS];
        attn_prepare_q(args, q, q_gain, rope, ok, kv, sg, lane, scratch, qv);
        float running_m;
        float running_l;
        float acc[BLOOMERY_ATTN_LANE_DIMS];
        attn_tile_loop(cache, args, ok, kv, seg, span, max_live, lane, qv, &running_m,
                       &running_l, acc);
        attn_epilogue(args, ATTN_GATE_SRC, partials, out, faults, ok, kv, seg, sg, lane, &running_m,
                      &running_l, acc);
    }

    // Past one segment the last threadgroup of each key head folds that key
    // head's partials; `ticket` holds the launch's word a key head, and each
    // fold resets its word for the next launch.
    if (args.segments > 1u && active) {
        ticket_is_last(ticket + kv, args.segments, tid, last_flag);
        if (last_flag == 1u) {
            attn_fold(partials, ticket + kv, args, ATTN_GATE_SRC, out, faults, tid, kv);
        }
    }
}

// A segmented flash at the segment cap launches KV x SEGMENT_CAP
// threadgroups, KV a core on a GPU of SEGMENT_CAP cores. The kernel declares
// KV threadgroups' threads its maximum, so the compiler keeps its registers
// within what holds them resident together and the dispatch runs as one wave.
// Past a threadgroup's thread limit (more than 32 query heads) no group size
// holds them, and the kernel declares none: the compiler's own allocation.
#if BLOOMERY_ATTN_THREADS * BLOOMERY_ATTN_KV <= 1024
#define ATTN_FLASH_HOLD [[max_total_threads_per_threadgroup(BLOOMERY_ATTN_THREADS * BLOOMERY_ATTN_KV)]]
#else
#define ATTN_FLASH_HOLD
#endif

// The threadgroup's one shared block, in 16-byte words: the scratch the
// append and the q prepare pair through.
ATTN_FLASH_HOLD
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
                             uint3 group [[threadgroup_position_in_grid]] ATTN_HGATE_PARAM(13)) {
    threadgroup uint4 shared[ATTN_SHARED_WORDS];
    threadgroup uint last_flag = 0u;
    AttnArgs args = launch;
    F16Cache cache;
    cache.k = cache_k;
    cache.v = cache_v;
    attn_flash<F16Cache, true>(cache, args, q, k, v, q_gain, k_gain, rope,
                               ATTN_HGATE_ACTUAL(0u) partials, out, faults, ticket, group.x,
                               tid, (threadgroup float *)shared, last_flag);
}

ATTN_FLASH_HOLD
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
                            uint3 group [[threadgroup_position_in_grid]] ATTN_HGATE_PARAM(15)) {
    threadgroup uint4 shared[ATTN_SHARED_WORDS];
    threadgroup uint last_flag = 0u;
    AttnArgs args = launch;
    Q8Cache cache;
    cache.k_codes = k_codes;
    cache.k_scales = k_scales;
    cache.v_codes = v_codes;
    cache.v_scales = v_scales;
    attn_flash<Q8Cache, true>(cache, args, q, k, v, q_gain, k_gain, rope,
                              ATTN_HGATE_ACTUAL(0u) partials, out, faults, ticket, group.x,
                              tid, (threadgroup float *)shared, last_flag);
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
#if BLOOMERY_ATTN_PAIRS_NORM
    // The prologue loads each pair as one word, so the scratch holds whole
    // pairs at a `float2`'s alignment.
    threadgroup float2 pairs[BLOOMERY_ATTN_HEAD / 2u];
    threadgroup float *const scratch = (threadgroup float *)pairs;
#else
    threadgroup float scratch[BLOOMERY_ATTN_HEAD];
#endif
    F16Cache cache;
    cache.k = cache_k;
    cache.v = cache_v;
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
// (key heads x segments, rows); every row has its own partials and its own
// ticket word a key head.
ATTN_FLASH_HOLD
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
                                  uint3 group [[threadgroup_position_in_grid]] ATTN_HGATE_PARAM(10)) {
    threadgroup uint4 shared[ATTN_SHARED_WORDS];
    threadgroup uint last_flag = 0u;
    F16Cache cache;
    cache.k = cache_k;
    cache.v = cache_v;
    uint row = group.y;
    AttnArgs a = attn_row_args(args, row);
    uint partials_stride = BLOOMERY_ATTN_Q * args.segments * BLOOMERY_ATTN_PARTIAL;
    attn_flash<F16Cache, false>(cache, a, q + row * args.q_stride, nullptr, nullptr, q_gain,
                                nullptr, rope, ATTN_HGATE_ACTUAL(row) partials + row * partials_stride,
                                out + row * args.out_stride, faults,
                                ticket + row * BLOOMERY_ATTN_KV, group.x, tid,
                                (threadgroup float *)shared, last_flag);
}

// The one-pass flash: the f16 launch of one segment. A threadgroup is one
// query head of BLOOMERY_ATTN_ONE_SG SIMD groups, and it reads every key the
// layer attends, so no threadgroup writes a partial and none folds. Key j of
// the window [lo, keys) belongs to group (j - lo) % ONE_SG; a group streams
// its keys from the cache into registers ATTN_ONE_CHUNK at a time, the lanes
// splitting the head as in the segmented flash, with that flash's tile
// arithmetic: the score is each lane's fmas in dim order, the SIMD group's
// sum and the scale; the softmax rescales once a chunk; a key past the live
// end scores -infinity and weighs exactly 0.
//
// Group 0 norms and ropes the query into threadgroup memory. Under APPEND,
// group 1 norms and ropes this position's key and group 2 takes its value,
// each rounded to the cache's halves into threadgroup memory: the group that
// holds the position reads that copy in place of the cache row, which is the
// same bits, and the threadgroup of the first query head of the key head
// stores both rows to the cache. No threadgroup of the dispatch reads the
// row the dispatch stores.
//
// The merge is one thread a value: the largest of the groups' maxima M, then
// each group's weight sum and value times exp(m - M), added by fma in
// ascending group order. A group that holds no live key has no weight and
// is skipped; a NaN maximum or weight sum writes NaN. The gate and the
// stores are the segmented epilogue's. The rows launch (APPEND false) reads
// every key from the cache its append dispatch wrote, so its row is the bits
// of the decode launch at that position.
#define ATTN_ONE_THREADS (BLOOMERY_ATTN_ONE_SG * BLOOMERY_ATTN_LANES)
#define ATTN_ONE_CHUNK BLOOMERY_ATTN_ONE_CHUNK

static_assert(BLOOMERY_ATTN_ONE_SG >= 3u, "the one-pass prepare takes three SIMD groups");
static_assert(ATTN_ONE_THREADS >= BLOOMERY_ATTN_HEAD, "the one-pass merge takes a thread a value");
static_assert(BLOOMERY_ATTN_ONE_SG * ATTN_ONE_CHUNK >= BLOOMERY_ATTN_TILE,
              "a round of chunks spans a tile: a key meets no more rescales than a tile's");

// The one-pass threadgroup's memory, in floats: the query, the key's norm
// scratch, the key's and the value's lane words (half a float a value), and
// each group's maximum, weight sum and values.
#define ATTN_ONE_FLOATS \
    (3u * BLOOMERY_ATTN_HEAD + BLOOMERY_ATTN_ONE_SG * (2u + BLOOMERY_ATTN_HEAD))

// A lane's values as the cache's lane word: each value `f32_to_f16_bits`,
// the first in the low half.
static attn_lane_word attn_pack_lane(thread float const row[BLOOMERY_ATTN_LANE_DIMS]) {
    uint w[BLOOMERY_ATTN_LANE_DIMS / 2u];
    for (uint i = 0u; i < BLOOMERY_ATTN_LANE_DIMS / 2u; ++i) {
        w[i] = uint(attn_f32_to_f16(row[2u * i])) | (uint(attn_f32_to_f16(row[2u * i + 1u])) << 16);
    }
#if BLOOMERY_ATTN_LANE_DIMS == 8
    return uint4(w[0], w[1], w[2], w[3]);
#else
    return uint2(w[0], w[1]);
#endif
}

template <bool APPEND>
static void attn_one(thread const AttnArgs &args,
                     device float const *q,
                     device float const *k,
                     device float const *v,
                     device float const *q_gain,
                     device float const *k_gain,
                     device float const *rope,
                     device float const *gate_src,
                     device ushort *cache_k,
                     device ushort *cache_v,
                     device float *out,
                     device atomic_uint *faults,
                     uint head,
                     uint tid,
                     threadgroup float *shared) {
    uint sg = tid / BLOOMERY_ATTN_LANES;
    uint lane = tid % BLOOMERY_ATTN_LANES;
    uint kv = head / BLOOMERY_ATTN_GQA;
    uint dim0 = lane * BLOOMERY_ATTN_LANE_DIMS;
    threadgroup float *sq = shared;
    threadgroup float *sk = sq + BLOOMERY_ATTN_HEAD;
    threadgroup attn_lane_word *kp = (threadgroup attn_lane_word *)(sk + BLOOMERY_ATTN_HEAD);
    threadgroup attn_lane_word *vp = kp + BLOOMERY_ATTN_LANES;
    threadgroup float *pm = (threadgroup float *)(vp + BLOOMERY_ATTN_LANES);
    threadgroup float *pl = pm + BLOOMERY_ATTN_ONE_SG;
    threadgroup float *pacc = pl + BLOOMERY_ATTN_ONE_SG;

    uint max_live;
    bool ok = attn_bounds(args, faults, true, tid, max_live);
    if (sg == 0u && ok) {
        for (uint d = 0u; d < BLOOMERY_ATTN_LANE_DIMS; ++d) {
            sq[dim0 + d] = q[head * BLOOMERY_ATTN_QG + dim0 + d];
        }
        float y[BLOOMERY_ATTN_LANE_DIMS];
        attn_norm_rope(sq, q_gain, rope + args.pos * BLOOMERY_ATTN_ROT, lane, args.eps, y);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint d = 0u; d < BLOOMERY_ATTN_LANE_DIMS; ++d) {
            sq[dim0 + d] = y[d];
        }
    }
    bool local = false;
    if constexpr (APPEND) {
        local = ok && args.pos < args.keys;
        bool stores = local && head % BLOOMERY_ATTN_GQA == 0u;
        uint row = kv * args.ctx + args.pos;
        if (sg == 1u && local) {
            device float const *rope_row = rope + args.pos * BLOOMERY_ATTN_ROT;
            for (uint d = 0u; d < BLOOMERY_ATTN_LANE_DIMS; ++d) {
                sk[dim0 + d] = k[kv * BLOOMERY_ATTN_HEAD + dim0 + d];
            }
            float y[BLOOMERY_ATTN_LANE_DIMS];
            attn_norm_rope(sk, k_gain, rope_row, lane, args.eps, y);
            attn_lane_word w = attn_pack_lane(y);
            kp[lane] = w;
            if (stores) {
                ((device attn_lane_word *)(cache_k + row * BLOOMERY_ATTN_HEAD))[lane] = w;
            }
        }
        if (sg == 2u && local) {
            float y[BLOOMERY_ATTN_LANE_DIMS];
            for (uint d = 0u; d < BLOOMERY_ATTN_LANE_DIMS; ++d) {
                y[d] = v[kv * BLOOMERY_ATTN_HEAD + dim0 + d];
            }
            attn_lane_word w = attn_pack_lane(y);
            vp[lane] = w;
            if (stores) {
                ((device attn_lane_word *)(cache_v + row * BLOOMERY_ATTN_HEAD))[lane] = w;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float qv[BLOOMERY_ATTN_LANE_DIMS];
    float acc[BLOOMERY_ATTN_LANE_DIMS];
    for (uint d = 0u; d < BLOOMERY_ATTN_LANE_DIMS; ++d) {
        qv[d] = sq[dim0 + d];
        acc[d] = 0.0f;
    }
    float m = -INFINITY;
    float l = 0.0f;
#if BLOOMERY_ATTN_WINDOW
    uint lo = attn_window_lo(args, max_live);
#else
    uint lo = 0u;
#endif
    if (ok) {
        const device attn_lane_word *ck =
            (const device attn_lane_word *)cache_k + kv * args.ctx * BLOOMERY_ATTN_LANES + lane;
        const device attn_lane_word *cv =
            (const device attn_lane_word *)cache_v + kv * args.ctx * BLOOMERY_ATTN_LANES + lane;
        for (uint base = lo + sg; base < max_live; base += BLOOMERY_ATTN_ONE_SG * ATTN_ONE_CHUNK) {
            attn_lane_word wk[ATTN_ONE_CHUNK];
            attn_lane_word wv[ATTN_ONE_CHUNK];
            for (uint c = 0u; c < ATTN_ONE_CHUNK; ++c) {
                uint j = base + c * BLOOMERY_ATTN_ONE_SG;
                attn_lane_word a = 0u;
                attn_lane_word b = 0u;
                if (j < max_live) {
                    if (APPEND && local && j == args.pos) {
                        a = kp[lane];
                        b = vp[lane];
                    } else {
                        a = ck[j * BLOOMERY_ATTN_LANES];
                        b = cv[j * BLOOMERY_ATTN_LANES];
                    }
                }
                wk[c] = a;
                wv[c] = b;
            }
            float s[ATTN_ONE_CHUNK];
            float chunk_max = -INFINITY;
            for (uint c = 0u; c < ATTN_ONE_CHUNK; ++c) {
                float kk[BLOOMERY_ATTN_LANE_DIMS];
                attn_widen(wk[c], kk);
                float dot = 0.0f;
                for (uint d = 0u; d < BLOOMERY_ATTN_LANE_DIMS; ++d) {
                    dot = fma(qv[d], kk[d], dot);
                }
                dot = simd_sum(dot);
                float score = dot * args.scale;
                s[c] = base + c * BLOOMERY_ATTN_ONE_SG < max_live ? score : -INFINITY;
                chunk_max = max(chunk_max, s[c]);
            }
            float new_m = max(m, chunk_max);
            float factor = m > -INFINITY ? exp(m - new_m) : 0.0f;
            l *= factor;
            for (uint d = 0u; d < BLOOMERY_ATTN_LANE_DIMS; ++d) {
                acc[d] *= factor;
            }
            m = new_m;
            for (uint c = 0u; c < ATTN_ONE_CHUNK; ++c) {
                float vv[BLOOMERY_ATTN_LANE_DIMS];
                attn_widen(wv[c], vv);
                float weight = base + c * BLOOMERY_ATTN_ONE_SG < max_live ? exp(s[c] - new_m) : 0.0f;
                l += weight;
                for (uint d = 0u; d < BLOOMERY_ATTN_LANE_DIMS; ++d) {
                    acc[d] = fma(weight, vv[d], acc[d]);
                }
            }
        }
    }
    if (lane == 0u) {
        pm[sg] = m;
        pl[sg] = l;
    }
    for (uint d = 0u; d < BLOOMERY_ATTN_LANE_DIMS; ++d) {
        pacc[sg * BLOOMERY_ATTN_HEAD + dim0 + d] = acc[d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid >= BLOOMERY_ATTN_HEAD) {
        return;
    }
    float big = -INFINITY;
    bool refused = false;
    for (uint g = 0u; g < BLOOMERY_ATTN_ONE_SG; ++g) {
        float mg = pm[g];
        float lg = pl[g];
        if (isnan(mg) || isnan(lg)) {
            refused = true;
        } else if (lg > 0.0f) {
            big = max(big, mg);
        }
    }
    float sum = 0.0f;
    float val = 0.0f;
    if (!refused) {
        for (uint g = 0u; g < BLOOMERY_ATTN_ONE_SG; ++g) {
            float lg = pl[g];
            if (lg > 0.0f) {
                float f = exp(pm[g] - big);
                sum = fma(lg, f, sum);
                val = fma(pacc[g * BLOOMERY_ATTN_HEAD + tid], f, val);
            }
        }
    }
    uint o = head * BLOOMERY_ATTN_HEAD + tid;
    if (!ok || refused || !(sum > 0.0f) || !isfinite(big)) {
        out[o] = NAN;
        return;
    }
    float attn = val * (1.0f / sum);
#if BLOOMERY_ATTN_GATED
    float g = gate_src[head * BLOOMERY_ATTN_QG + BLOOMERY_ATTN_HEAD + tid];
#elif BLOOMERY_ATTN_HEAD_GATE
    float g = gate_src[head];
#endif
#if ATTN_GATE
    attn_commit(out, o, attn, g, attn * attn_sigmoid(g), faults, args.layer);
#else
    attn_commit_plain(out, o, attn);
#endif
}

// The one-pass decode launch: one threadgroup a query head. The bindings are
// gqa256_flash_f16's; the partials and the ticket are bound and not read.
[[kernel, max_total_threads_per_threadgroup(ATTN_ONE_THREADS)]]
void attn_flash_one_f16(device float const *q [[buffer(0)]],
                        device float const *k [[buffer(1)]],
                        device float const *v [[buffer(2)]],
                        device float const *q_gain [[buffer(3)]],
                        device float const *k_gain [[buffer(4)]],
                        device float const *rope [[buffer(5)]],
                        device ushort *cache_k [[buffer(6)]],
                        device ushort *cache_v [[buffer(7)]],
                        device float *out [[buffer(9)]],
                        constant AttnArgs &launch [[buffer(10)]],
                        device atomic_uint *faults [[buffer(11)]],
                        uint tid [[thread_index_in_threadgroup]],
                        uint3 group [[threadgroup_position_in_grid]] ATTN_HGATE_PARAM(13)) {
    threadgroup uint4 shared[(ATTN_ONE_FLOATS + 3u) / 4u];
    AttnArgs args = launch;
    attn_one<true>(args, q, k, v, q_gain, k_gain, rope, ATTN_GATE_SRC, cache_k, cache_v, out,
                   faults, group.x, tid, (threadgroup float *)shared);
}

// The one-pass rows launch: gqa256_flash_f16_rows's bindings and row grid
// dimension, one threadgroup a (query head, row).
[[kernel, max_total_threads_per_threadgroup(ATTN_ONE_THREADS)]]
void attn_flash_one_f16_rows(device float const *q [[buffer(0)]],
                             device float const *q_gain [[buffer(1)]],
                             device float const *rope [[buffer(2)]],
                             device ushort *cache_k [[buffer(3)]],
                             device ushort *cache_v [[buffer(4)]],
                             device float *out [[buffer(6)]],
                             constant AttnRowsArgs &args [[buffer(7)]],
                             device atomic_uint *faults [[buffer(8)]],
                             uint tid [[thread_index_in_threadgroup]],
                             uint3 group [[threadgroup_position_in_grid]] ATTN_HGATE_PARAM(10)) {
    threadgroup uint4 shared[(ATTN_ONE_FLOATS + 3u) / 4u];
    uint row = group.y;
    AttnArgs a = attn_row_args(args, row);
#if BLOOMERY_ATTN_HEAD_GATE
    device float const *gate_src = hgate + (ulong)row * BLOOMERY_ATTN_Q;
#else
    device float const *gate_src = q + row * args.q_stride;
#endif
    attn_one<false>(a, q + row * args.q_stride, nullptr, nullptr, q_gain, nullptr, rope, gate_src,
                    cache_k, cache_v, out + row * args.out_stride, faults, group.x, tid,
                    (threadgroup float *)shared);
}

// The flash of a chunk of rows with simdgroup matrices. A threadgroup is four
// SIMD groups over one key head. The key head's rows are its query heads at
// every position of the chunk, numbered position by position: row n is query
// head n % GQA at chunk row n / GQA. A threadgroup takes 32 consecutive rows
// and each SIMD group the 8 of one fragment, so every row of a fragment is
// real: a position's heads may fill a fragment, share one with the next
// position, or straddle two SIMD groups, and a row owns nothing but its own
// query, its running max and weight sum and its values, which stay in the
// SIMD group's registers. The 8x8 fragments of the score matrix are (rows x
// keys), so every key of a tile serves all 32 rows. A lane holds two columns
// of one row, so the causal mask is its row's position and the window its
// row's. The K and V tiles of 16 keys sit in threadgroup memory beside the
// other three SIMD groups' and are the loop's only shared state. A threadgroup
// reads the tiles up to its last row's position; a tile past an earlier row's
// own position is an exact no-op for it.
//
// The queries are normed and roped by the decode's own functions into a
// threadgroup stage, rounded to f16, and held as fragments for the whole
// loop. The score is the fragment product of f16 operands in f32 over the
// head's dims, times the decode's scale. The softmax is online in f32 over tiles of
// 16 keys taken from absolute key 0, so a row's tiles do not depend on the
// chunk it came in or on the rows around it; the weights of a tile are
// rounded to f16 for the value product, and the weight sum is the f32
// weights'. A tile beyond a row's own position is an exact no-op for the
// row: its weights are 0 and its rescale factor 1. The rescale is the row's
// own factor on its accumulators, one multiply a value, so no row reads
// another's. The sums are therefore not the decode flash's: the rows are
// held to the decode flash by a band, not by bits.
//
// A query value past f16's range raises the product site and its row is NaN.
// A key or value row that is not finite makes NaN the rows of its tile and
// after, and raises nothing, as in the decode flash. The tail of the chunk
// repeats its last row in the rows past it, and stores nothing for them.
#include <metal_simdgroup_matrix>

#define PF_STRIDE (BLOOMERY_ATTN_HEAD + BLOOMERY_ATTN_PF_PAD)
#define PF_TILE_HALVES (BLOOMERY_ATTN_PF_KEYS * PF_STRIDE)
// The rows of a fragment, and of a threadgroup's four.
#define PF_ROWS 8u
#define PF_BLOCK_ROWS (BLOOMERY_ATTN_PF_SGS * PF_ROWS)
// A key head's row n: the chunk row it sits at and the query head it is.
#define PF_ROW_OF(n) ((n) / BLOOMERY_ATTN_GQA)
#define PF_HEAD_OF(n) ((n) % BLOOMERY_ATTN_GQA)
// Every loop over the fragments of the query or the accumulator is unrolled
// in full: a fragment array indexed by a run-time value lives in memory, and
// the products then run at a fraction of the matrix unit's rate.
#define PF_DCHUNKS (BLOOMERY_ATTN_HEAD / 8u)
// A cache row in 16-byte words, and the loads a thread makes to fill a tile.
#define PF_ROW_WORDS (BLOOMERY_ATTN_HEAD / 8u)
#define PF_LOADS (BLOOMERY_ATTN_PF_KEYS * PF_ROW_WORDS / (BLOOMERY_ATTN_PF_SGS * 32u))

static_assert(PF_BLOCK_ROWS * PF_STRIDE <= 2u * PF_TILE_HALVES,
              "the query stage fits in the K and V tiles it shares memory with");
static_assert(BLOOMERY_ATTN_PF_KEYS == 16u, "a tile is two 8-key fragments");
static_assert(PF_LOADS * BLOOMERY_ATTN_PF_SGS * 32u == BLOOMERY_ATTN_PF_KEYS * PF_ROW_WORDS,
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
#if BLOOMERY_ATTN_PAIRS_NORM
    // The prologue loads each pair as one word, so the stage holds whole
    // pairs at a `float2`'s alignment.
    threadgroup float2 stage[BLOOMERY_ATTN_PF_SGS * BLOOMERY_ATTN_HEAD / 2u];
    threadgroup float *const scratch = (threadgroup float *)stage;
#else
    threadgroup float scratch[BLOOMERY_ATTN_PF_SGS * BLOOMERY_ATTN_HEAD];
#endif
    threadgroup half *tile_k = shared;
    threadgroup half *tile_v = shared + PF_TILE_HALVES;

    // Where this lane sits in a fragment: row `fm` of the SIMD group's eight,
    // columns `fn` and `fn + 1`.
    ushort qid = ushort(lane / 4u);
    ushort fm = (qid & 4u) + ((lane / 2u) % 4u);
    ushort fn = (qid & 2u) * 2u + (lane % 2u) * 2u;

    // The heaviest blocks (the last positions) are the first threadgroups.
    uint kv = group.y;
    uint total = args.rows * BLOOMERY_ATTN_GQA;
    uint n0 = (tpg.x - 1u - group.x) * PF_BLOCK_ROWS;
    uint last = PF_ROW_OF(min(n0 + PF_BLOCK_ROWS, total) - 1u);
    // The SIMD group's rows; those past the chunk repeat its last row.
    uint sg_n0 = n0 + sg * PF_ROWS;
    bool writes = sg_n0 < total;
    AttnArgs bound = attn_row_args(args, PF_ROW_OF(min(sg_n0 + PF_ROWS, total) - 1u));
    uint max_live;
    bool ok = attn_bounds(bound, faults, writes, lane, max_live);
    uint block_keys = args.from + last + 1u;
    // A block past the cache has no key to read; its rows are the bounds' NaN.
    if (block_keys > args.ctx) {
        block_keys = 0u;
    }
    // This lane's row: the chunk row it sits at, and that row's words.
    uint n = min(sg_n0 + fm, total - 1u);
    uint row = PF_ROW_OF(n);
    AttnArgs a = attn_row_args(args, row);
#if BLOOMERY_ATTN_WINDOW
    // The first key this row attends, and the first tile of the block: the one
    // holding the window of its first row, the earliest of the block's.
    AttnArgs first = attn_row_args(args, PF_ROW_OF(n0));
    uint row_lo = attn_window_lo(a, a.keys);
    uint first_tile = attn_window_lo(first, first.keys) / BLOOMERY_ATTN_PF_KEYS;
#define ATTN_PF_FIRST_TILE first_tile
#else
#define ATTN_PF_FIRST_TILE 0u
#endif

    // The queries of the SIMD group's eight rows, normed and roped, as f16 rows
    // of the stage.
    threadgroup float *sc = scratch + sg * BLOOMERY_ATTN_HEAD;
    bool overflow = false;
    uint bad_rows = 0u;
    for (uint r = 0u; r < PF_ROWS; ++r) {
        uint nr = min(sg_n0 + r, total - 1u);
        uint qrow = PF_ROW_OF(nr);
        uint head = kv * BLOOMERY_ATTN_GQA + PF_HEAD_OF(nr);
        float y[BLOOMERY_ATTN_LANE_DIMS];
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (ok) {
            device float const *src = q + (ulong)qrow * args.q_stride + head * BLOOMERY_ATTN_QG +
                                      lane * BLOOMERY_ATTN_LANE_DIMS;
            for (uint d = 0u; d < BLOOMERY_ATTN_LANE_DIMS; ++d) {
                sc[lane * BLOOMERY_ATTN_LANE_DIMS + d] = src[d];
            }
            attn_norm_rope(sc, q_gain, rope + (args.from + qrow) * BLOOMERY_ATTN_ROT, lane, a.eps, y);
        } else {
            for (uint d = 0u; d < BLOOMERY_ATTN_LANE_DIMS; ++d) {
                y[d] = 0.0f;
            }
        }
        threadgroup half *dst = shared + (sg * PF_ROWS + r) * PF_STRIDE +
                                lane * BLOOMERY_ATTN_LANE_DIMS;
        bool bad = false;
        for (uint d = 0u; d < BLOOMERY_ATTN_LANE_DIMS; ++d) {
            half hv = half(y[d]);
            if (!isfinite(float(hv))) {
                bad = true;
                overflow = overflow || isfinite(y[d]);
            }
            dst[d] = hv;
        }
        bad_rows |= uint(simd_any(bad)) << r;
    }
    // A row whose query is unusable, or any row of a group the bounds refuse,
    // is NaN.
    bool unusable = !ok || ((bad_rows >> fm) & 1u) != 0u;
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

    simdgroup_float8x8 acc[PF_DCHUNKS];
    #pragma clang loop unroll(full)
    for (uint c = 0u; c < PF_DCHUNKS; ++c) {
        acc[c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    float running_m = -INFINITY;
    float running_l = 0.0f;

    uint tiles = (block_keys + BLOOMERY_ATTN_PF_KEYS - 1u) / BLOOMERY_ATTN_PF_KEYS;
    for (uint t = ATTN_PF_FIRST_TILE; t < tiles; ++t) {
        uint k0 = t * BLOOMERY_ATTN_PF_KEYS;
        for (uint i = 0u; i < PF_LOADS; ++i) {
            uint e = tid + i * (BLOOMERY_ATTN_PF_SGS * 32u);
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
        // The row's rescale on the accumulators: the lane holds two values of
        // its row in each fragment.
        #pragma clang loop unroll(full)
    for (uint c = 0u; c < PF_DCHUNKS; ++c) {
            acc[c].thread_elements()[0] *= factor;
            acc[c].thread_elements()[1] *= factor;
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
    if (sg_n0 + fm >= total) {
        return;
    }
    uint head = kv * BLOOMERY_ATTN_GQA + PF_HEAD_OF(n);
    bool whole = !unusable && running_l > 0.0f && isfinite(running_m);
    float inv = 1.0f / running_l;
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
#if BLOOMERY_ATTN_GATED
            float g = q[(ulong)row * args.q_stride + head * BLOOMERY_ATTN_QG + BLOOMERY_ATTN_HEAD + dim];
#elif BLOOMERY_ATTN_HEAD_GATE
            float g = hgate[(ulong)row * BLOOMERY_ATTN_Q + head];
#endif
#if ATTN_GATE
            float product = attn * attn_sigmoid(g);
            attn_commit(out, at, attn, g, product, faults, a.layer);
#else
            attn_commit_plain(out, at, attn);
#endif
        }
    }
}
