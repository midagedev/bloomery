// The relative-position bias attention of Inkling 2, one visit: the
// segment walk of the decode attention's own structure at the group of
// three (BLOOMERY_INK2_Q over BLOOMERY_INK2_KV, one SIMD group a query
// head, BLOOMERY_INK2_HEAD values a head, four a lane), with the two facts
// of the bias laid in:
//
// - the score of key j gains b_j = Σ_{d < REL} rel_bias[d, dist_j] · r_d,
//   dist_j = live − 1 − key position, the key positions of a tile being
//   consecutive so the tile's [REL][TILE] slice of the table is staged in
//   threadgroup memory beside the walk. A distance at or past
//   BLOOMERY_INK2_EXTENT reads 0 — the reference's pad column, which the
//   caller does not upload;
// - the softmax denominator gains a zero column: the per-(head, query) sink
//   logit t joins the fold's partial as one more logit in the denominator
//   and nothing in the value, exp(s − t) / (1 + Σ exp(s_k − t)) — the
//   reference's distribution, the rounding points its own gate's band.
//
// No rope runs here at all: nothing turns a value (the entries never read a
// rope table), and the q, k, v and r rows arrive already clipped and
// appended by the epilogue (qk_clip.metal). The visit's cache planes are
// f16, head-major [KV][rows][HEAD] u16, a position's row position mod rows
// — a plane passes its context as rows (the modulo is the identity), a
// windowed ring its own row count.
//
// The walk reads the keys and values a tile of the attention's own TILE at
// a time, each lane its 8-byte word of a row straight from the cache into
// registers, the online softmax rescaling once a tile; the score is the
// lane's fmas in dim order, the SIMD group's xor butterfly (the attention's
// own), the scale, then the bias dot — REL fused multiply-adds ascending d
// against the staged tile, the r row in registers. The segments take the
// attention's own span rule (attn_span); past its one-pass waterline a
// window of 1024 keys splits at the segment cap, and each key head's last
// threadgroup folds its partials and the sink — no merge dispatch.
//
// No silent failure: a live count of zero or past the context raises the
// key-count site and the row is NaN; a sink that is NaN or +inf has no
// defined softmax and writes NaN into the row (a −inf sink is neutral).
//
// This file comes after fault.metal (raise), attention.metal (attn_span,
// the tile and span rule the walk shares) and ticket.metal.

#pragma METAL fp contract(off)

#define REL_Q BLOOMERY_INK2_Q
#define REL_KV BLOOMERY_INK2_KV
#define REL_HEAD BLOOMERY_INK2_HEAD
#define REL_GROUP (REL_Q / REL_KV)
#define REL_DIMS (REL_HEAD / 32u)
#define REL_TILE BLOOMERY_ATTN_TILE
#define REL_WALK_AHEAD 4u
#define REL_SEGMENTS BLOOMERY_ATTN_SEGMENT_CAP
#define REL_REL BLOOMERY_INK2_REL

static_assert(REL_GROUP * REL_KV == REL_Q, "the group covers the query heads");
static_assert(REL_GROUP * 32u <= 1024u, "the flash's threadgroup fits the device's threads");
static_assert(REL_DIMS == 4u, "a lane holds four values of a head of 128");
static_assert(REL_TILE == 16u && REL_WALK_AHEAD == 4u,
              "the walk's butterfly halves four rows over 32 lanes");
static_assert(REL_SEGMENTS <= 32u, "a folding lane holds its segments in one round");
static_assert(REL_TILE <= 32u, "a lane keeps one tile key's score");

// The launch's argument block, bound by value.
struct RelFlashArgs {
    uint segments; // the segments the launch's keys take
    uint ctx;      // the context the live count is bounded by
    uint rows;     // the cache planes' row count (a ring's own, a plane's the context)
    uint keys;     // live keys, this position's own among them
    uint window;   // the last window of the live keys attended, 0 for all
    uint layer;
    float scale;   // the score's scale, 1/sqrt(head) the caller derives
};

static_assert(sizeof(RelFlashArgs) == 28, "RelFlashArgs is 28 bytes: the host packs seven words");

// The first key a visit attends: the last `window` of its `live` keys, the
// first of them when the launch names no window — the decode attention's
// own rule, stated here because its kernel guards its copy behind the
// window axis.
inline uint rel_window_lo(thread const RelFlashArgs &args, uint live) {
    return (args.window != 0u && live > args.window) ? live - args.window : 0u;
}

// A lane's 8-byte word of halves widened by the hardware conversion (the
// attention's own at four values a lane).
inline void rel_widen(uint2 w, thread float out[REL_DIMS]) {
    const float4 f = float4(as_type<half4>(w));
    out[0] = f.x;
    out[1] = f.y;
    out[2] = f.z;
    out[3] = f.w;
}

// The bias tile of one key tile, staged: thread tid's share of the
// REL x TILE entries, column key holding the table column of the key's
// distance, 0 past the extent (the pad rule) and 0 for a key the visit does
// not hold.
static void rel_stage_bias(device const float *bias,
                           threadgroup float *tile,
                           uint live,
                           uint lo,
                           uint hi,
                           uint tid)
{
    const uint threads = REL_GROUP * 32u;
    for (uint e = tid; e < REL_REL * REL_TILE; e += threads) {
        const uint d = e / REL_TILE;
        const uint key = e - d * REL_TILE;
        float val = 0.0f;
        if (lo + key < hi) {
            const uint dist = live - 1u - (lo + key);
            if (dist < BLOOMERY_INK2_EXTENT) {
                val = bias[d * BLOOMERY_INK2_EXTENT + dist];
            }
        }
        tile[e] = val;
    }
}

// One tile of the walk: the keys' scores and weights, the running softmax
// rescaled once. Lane t keeps the score of the tile's key t & 15; the
// score is the lane's dot by the butterfly, the scale, then the bias dot.
template <bool FULL>
static void rel_tile(device const ushort *cache_k,
                     device const ushort *cache_v,
                     thread const RelFlashArgs &args,
                     threadgroup float *bias_tile,
                     thread float const qr[REL_DIMS],
                     thread float const rr[REL_REL],
                     uint live,
                     uint row0,
                     uint lo,
                     uint hi,
                     uint lane,
                     thread float *running_m,
                     thread float *running_l,
                     thread float acc[REL_DIMS])
{
    const bool b0 = (lane & 1u) != 0u;
    const bool b1 = (lane & 2u) != 0u;
    float quads[REL_TILE / REL_WALK_AHEAD];
    for (uint t0 = 0u; t0 < REL_TILE; t0 += REL_WALK_AHEAD) {
        uint2 words[REL_WALK_AHEAD];
        for (uint c = 0u; c < REL_WALK_AHEAD; ++c) {
            const uint key = lo + t0 + c;
            words[c] = FULL || key < hi
                           ? ((device const uint2 *)(cache_k +
                                                     (row0 + key % args.rows) * REL_HEAD))[0]
                           : uint2(0u);
        }
        float part[REL_WALK_AHEAD];
        for (uint c = 0u; c < REL_WALK_AHEAD; ++c) {
            float kv[REL_DIMS];
            rel_widen(words[c], kv);
            float dot = 0.f;
            for (uint d = 0u; d < REL_DIMS; ++d) {
                dot = fma(qr[d], kv[d], dot);
            }
            part[c] = dot;
        }
        const float pair0 =
            (b0 ? part[1] : part[0]) + simd_shuffle_xor(b0 ? part[0] : part[1], ushort(1));
        const float pair1 =
            (b0 ? part[3] : part[2]) + simd_shuffle_xor(b0 ? part[2] : part[3], ushort(1));
        quads[t0 / REL_WALK_AHEAD] =
            (b1 ? pair1 : pair0) + simd_shuffle_xor(b1 ? pair0 : pair1, ushort(2));
    }
    const bool b2 = (lane & 4u) != 0u;
    const bool b3 = (lane & 8u) != 0u;
    const float half0 =
        (b2 ? quads[1] : quads[0]) + simd_shuffle_xor(b2 ? quads[0] : quads[1], ushort(4));
    const float half1 =
        (b2 ? quads[3] : quads[2]) + simd_shuffle_xor(b2 ? quads[2] : quads[3], ushort(4));
    const float sixteen =
        (b3 ? half1 : half0) + simd_shuffle_xor(b3 ? half0 : half1, ushort(8));
    // The dot by the butterfly, the scale, then the bias dot — REL fused
    // multiply-adds ascending d against the staged tile; this order is the
    // gate.
    const float scaled =
        (sixteen + simd_shuffle_xor(sixteen, ushort(16))) * args.scale;
    const uint t = lane & (REL_TILE - 1u);
    const float bias_dot = fma(rr[0], bias_tile[t], 0.0f);
    float mine = -INFINITY;
    if (FULL || lo + t < hi) {
        float b = bias_dot;
        for (uint d = 1u; d < REL_REL; ++d) {
            b = fma(rr[d], bias_tile[d * REL_TILE + t], b);
        }
        mine = scaled + b;
    }
    const float new_m = max(*running_m, simd_max(mine));
    const float old_m = *running_m;
    const float factor = old_m > -INFINITY ? exp(old_m - new_m) : 0.0f;
    const float weight = lo + lane < hi ? exp(mine - new_m) : 0.0f;
    *running_l = *running_l * factor + simd_sum(weight);
    for (uint d = 0u; d < REL_DIMS; ++d) {
        acc[d] *= factor;
    }
    *running_m = new_m;
    for (uint t0 = 0u; t0 < REL_TILE; t0 += REL_WALK_AHEAD) {
        for (uint c = 0u; c < REL_WALK_AHEAD; ++c) {
            const uint key = lo + t0 + c;
            const uint2 word =
                FULL || key < hi
                    ? ((device const uint2 *)(cache_v + (row0 + key % args.rows) * REL_HEAD))[0]
                    : uint2(0u);
            const float w = simd_shuffle(weight, ushort(t0 + c));
            float kv[REL_DIMS];
            rel_widen(word, kv);
            for (uint d = 0u; d < REL_DIMS; ++d) {
                acc[d] = fma(w, kv[d], acc[d]);
            }
        }
    }
}

[[max_total_threads_per_threadgroup(REL_GROUP * 32u)]]
kernel void rel_flash_f16(device float const *q [[buffer(0)]],
                          device ushort *cache_k [[buffer(1)]],
                          device ushort *cache_v [[buffer(2)]],
                          device float const *r [[buffer(3)]],
                          device float const *bias [[buffer(4)]],
                          device coherent(device) float *partials [[buffer(5)]],
                          device float *out [[buffer(6)]],
                          constant RelFlashArgs &args [[buffer(7)]],
                          device atomic_uint *faults [[buffer(8)]],
                          uint tid [[thread_index_in_threadgroup]],
                          uint group [[threadgroup_position_in_grid]])
{
    RelFlashArgs a = args;
    const uint sg = tid / 32u;
    const uint lane = tid % 32u;
    const uint kv = group / a.segments;
    const uint seg = group - kv * a.segments;
    if (kv >= REL_KV || seg >= a.segments) {
        return;
    }
    const bool refused = a.keys == 0u || a.keys > a.ctx;
    if (tid == 0u && refused) {
        raise(faults, a.layer, BLOOMERY_FAULT_SITE_KEY_COUNT);
    }
    const uint live = refused ? 0u : a.keys;
    const uint lo_all = rel_window_lo(a, live);
    const uint span = attn_span(live - lo_all, a.segments);
    const uint lo = lo_all + seg * span;
    const uint hi = min(lo_all + (seg + 1u) * span, live);
    const uint head = kv * REL_GROUP + sg;

    // This group's query head and its relative row, in registers.
    float qr[REL_DIMS];
    for (uint d = 0u; d < REL_DIMS; ++d) {
        qr[d] = q[head * REL_HEAD + lane * REL_DIMS + d];
    }
    float rr[REL_REL];
    for (uint d = 0u; d < REL_REL; ++d) {
        rr[d] = r[head * REL_REL + d];
    }

    threadgroup float bias_tile[REL_REL][REL_TILE];
    float running_m = -INFINITY;
    float running_l = 0.f;
    float acc[REL_DIMS];
    for (uint d = 0u; d < REL_DIMS; ++d) {
        acc[d] = 0.f;
    }
    const uint row0 = kv * a.rows;
    if (!refused && lo < hi) {
        for (uint t0 = lo; t0 < hi; t0 += REL_TILE) {
            const uint tile_hi = min(t0 + REL_TILE, hi);
            rel_stage_bias(bias, &bias_tile[0][0], live, t0, tile_hi, tid);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tile_hi - t0 == REL_TILE) {
                rel_tile<true>(cache_k, cache_v, a, &bias_tile[0][0], qr, rr, live, row0, t0,
                               tile_hi, lane, &running_m, &running_l, acc);
            } else {
                rel_tile<false>(cache_k, cache_v, a, &bias_tile[0][0], qr, rr, live, row0, t0,
                                tile_hi, lane, &running_m, &running_l, acc);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // The partial, one a (head, segment): the max, the weight sum, the
    // head. An empty segment carries no weight and the fold skips it.
    const uint base = (head * a.segments + seg) * (REL_HEAD + 2u);
    if (refused || lo >= hi) {
        if (lane == 0u) {
            partials[base] = -INFINITY;
            partials[base + 1u] = 0.0f;
        }
    } else {
        if (lane == 0u) {
            partials[base] = running_m;
            partials[base + 1u] = running_l;
        }
        for (uint d = 0u; d < REL_DIMS; ++d) {
            partials[base + 2u + lane * REL_DIMS + d] = acc[d];
        }
    }
}

// The fold: one threadgroup a key head, its SIMD groups the key head's
// query heads. Lane s holds segment s of the head (the cap is one round of
// 32), folds its partials ascending, then the head's sink joins as one more
// logit in the denominator and nothing in the value, and the row is
// acc · (1 / l). The fold reads the partials the flash wrote through the
// threadgroup's device fence, so the launch runs it in a dispatch of its
// own after the flash's.
[[max_total_threads_per_threadgroup(REL_GROUP * 32u)]]
kernel void rel_flash_fold(device coherent(device) float *partials [[buffer(0)]],
                           device float const *sinks [[buffer(1)]],
                           device float *out [[buffer(2)]],
                           constant RelFlashArgs &args [[buffer(3)]],
                           device atomic_uint *faults [[buffer(4)]],
                           uint tid [[thread_index_in_threadgroup]],
                           uint group [[threadgroup_position_in_grid]])
{
    RelFlashArgs a = args;
    const uint sg = tid / 32u;
    const uint lane = tid % 32u;
    const uint kv = group;
    if (kv >= REL_KV) {
        return;
    }
    const uint head = kv * REL_GROUP + sg;
    const uint slot = lane;
    float mj = -INFINITY;
    float lj = 0.f;
    if (slot < a.segments) {
        const uint base = (head * a.segments + slot) * (REL_HEAD + 2u);
        mj = partials[base];
        lj = partials[base + 1u];
    }
    // The head's maximum over the segments that hold weight, each lane's
    // factor, and the weight sum.
    const bool carries = lj > 0.f && isfinite(mj);
    float big = carries ? mj : -INFINITY;
    big = simd_max(big);
    const float factor = carries && big > -INFINITY ? exp(mj - big) : 0.f;
    const float weights = simd_sum(carries ? lj * factor : 0.f);
    // The value sum: the segments' values added by the SIMD group, a dim at
    // a time, each lane its segment's partial times its factor.
    float acc[REL_DIMS];
    for (uint d = 0u; d < REL_DIMS; ++d) {
        float own = 0.f;
        if (slot < a.segments && carries && big > -INFINITY) {
            const uint at = (head * a.segments + slot) * (REL_HEAD + 2u) + 2u;
            own = partials[at + lane * REL_DIMS + d] * factor;
        }
        acc[d] = simd_sum(own);
    }
    // The sink joins last, after every live segment: one more logit in the
    // denominator, nothing in the value. A sink that is NaN or +inf has no
    // defined softmax (the row is NaN); a −inf one is neutral.
    const float sink = sinks[head];
    const bool refused = a.keys == 0u || a.keys > a.ctx;
    const bool bad = refused || isnan(sink) || sink == INFINITY;
    const float m = max(big, sink);
    const float rescale = big > -INFINITY ? exp(big - m) : 0.f;
    const float l = weights * rescale + (sink > -INFINITY ? exp(sink - m) : 0.f);
    for (uint d = 0u; d < REL_DIMS; ++d) {
        const uint at = head * REL_HEAD + lane * REL_DIMS + d;
        if (bad || !(l > 0.f) || !isfinite(m)) {
            out[at] = NAN;
        } else {
            out[at] = acc[d] * rescale / l;
        }
    }
}
