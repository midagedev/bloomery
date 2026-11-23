// The prompt pass's own kernels of Inkling 2, each the step kernel's rule
// over a chunk of rows:
//
// - qk_clip_append_rows: qk_clip_append (qk_clip.metal) at every row of the
//   chunk, row t at position pos + t, its key and value appended at plane
//   row (pos + t) mod ctx (a windowed visit's ring, a windowless visit's
//   plane), its q and r carrying taus[t] on a windowless visit. The row's
//   arithmetic is the step's, term for term.
// - rel_prefill_f16: the rel-bias attention (rel_flash.metal) over the
//   chunk's rows with simdgroup matrices, attention.metal's attn_prefill_f16
//   structure at the group of three: a threadgroup is four SIMD groups over
//   one key head, the head's rows numbered position by position (row n is
//   query head n % 3 at chunk row n / 3), 32 consecutive rows a threadgroup,
//   8 a SIMD group. The score of key j gains the bias dot over the relative
//   row, Σ_d r_d · rel_bias[d, p − j] (0 at a distance at or past the
//   extent), REL fused multiply-adds ascending d as the step's; the softmax
//   is online in f32 over 16-key tiles from absolute key 0 (or the block's
//   first window tile), and the head's sink joins at the end as one more
//   logit in the denominator. The products are f16 operands summed in f32
//   (q, the keys, the tile's weights each rounded to half), so a row is held
//   to the step's attention by a band, not by bits.
// - router_sigmoid_rows: router_sigmoid (router_situ.metal) a row a
//   threadgroup: the same dots (a lane's float4s, the four products added
//   left to right, the group's lanes summed), the same sigmoid and the same
//   selection, so a row whose input is the step's bits picks the step's ids
//   and weights bit for bit.
// - ink2_route_table: route_table (route.metal) over SI_EXPERTS groups and
//   no shared slot: the instance's router row carries a gated shared group
//   the family does not have, so the table the grouped tiles walk counts,
//   scans and places the picks alone, in route_table's table format.
// - combine_rows_store: combine_rows (route.metal) that stores the slots'
//   sum instead of adding it, the sum from zero in slot order as there: the
//   step's sum row starts at zero, so the bits are the step's.
//
// No silent failure: each kernel raises what its step kernel raises (the
// clip's cache-value and quant-column sites, the attention's key-count and
// product sites, the router's site); the host refuses a chunk past the
// cache or a ring narrower than the window plus the chunk's rows less one.
//
// This file comes after fault.metal (raise), norm.metal (rms_scale,
// rms_usable), attention.metal (attn_f32_to_f16), qk_clip.metal
// (qk_clip_factor, qk_clip_half), router.metal (router_load4), ticket.metal
// (argmax_key), router_situ.metal (SigmoidArgs and its constants) and
// route.metal (CombineArgs, RouteTableArgs, ROUTE_TILE, ROUTE_THREADS).

#include <metal_simdgroup_matrix>

#pragma METAL fp contract(off)

// ---------------------------------------------------------------------------
// The clip and append over rows.

struct QkClipRowsArgs {
    uint ctx;       // the planes' rows: a ring's own, a plane's the context
    uint pos;       // the chunk's first position
    uint layer;
    float v_eps;
    float qk_thresh;
    float r_thresh;
    uint taus;      // 1: row t's q and r carry taus[t]; 0: they carry 1.0
    uint qr_stride; // elements between rows of q (and of r)
    uint kv_stride; // elements between rows of k and of v
    uint rows;      // the chunk's rows
    uint r_at;      // a row's relative values, past its query values
    uint pad;
};

static_assert(sizeof(QkClipRowsArgs) == 48, "QkClipRowsArgs is twelve words");

// q holds the rows of the query values with each row's relative values at
// r_at past its start: the q path's one buffer, clipped in place.
kernel void qk_clip_append_rows(device float *q [[buffer(0)]],
                                device float *k [[buffer(1)]],
                                device float *v [[buffer(2)]],
                                device const float *gq [[buffer(3)]],
                                device const float *gr [[buffer(4)]],
                                device ushort *cache_k [[buffer(5)]],
                                device ushort *cache_v [[buffer(6)]],
                                constant QkClipRowsArgs &args [[buffer(7)]],
                                device atomic_uint *faults [[buffer(8)]],
                                device const float *taus [[buffer(9)]],
                                uint lane [[thread_index_in_simdgroup]],
                                uint sg [[simdgroup_index_in_threadgroup]],
                                uint3 group [[threadgroup_position_in_grid]])
{
    const uint heads = BLOOMERY_INK2_Q;
    const uint kv = BLOOMERY_INK2_KV;
    const uint gw = group.x * 8u + sg;
    const uint t = group.y;
    if (gw >= 2u * (heads + kv) || t >= args.rows) {
        return;
    }
    q += (ulong)t * args.qr_stride;
    device float *r = q + args.r_at;
    k += (ulong)t * args.kv_stride;
    v += (ulong)t * args.kv_stride;
    const uint slot = (args.pos + t) % args.ctx;
    const float tau = args.taus != 0u ? taus[t] : 1.0f;

    if (gw < heads + kv) {
        const bool is_q = gw < heads;
        const uint h = is_q ? gw : gw - heads;
        const uint base = h * BLOOMERY_INK2_HEAD;
        device float *rowbuf = is_q ? q : k;
        float x[4];
        float acc = 0.f;
        for (uint i = 0u; i < 4u; ++i) {
            x[i] = rowbuf[base + lane + 32u * i];
            if (is_q) {
                x[i] = x[i] * gq[base + lane + 32u * i];
            }
            acc = fma(x[i], x[i], acc);
        }
        const float f = qk_clip_factor(sqrt(simd_sum(acc)), args.qk_thresh);
        if (!isfinite(f)) {
            if (lane == 0u) {
                raise(faults, args.layer, BLOOMERY_FAULT_SITE_CACHE_VALUE);
            }
            for (uint i = 0u; i < 4u; ++i) {
                rowbuf[base + lane + 32u * i] = NAN;
            }
            return;
        }
        for (uint i = 0u; i < 4u; ++i) {
            rowbuf[base + lane + 32u * i] = is_q ? x[i] * f * tau : x[i] * f;
        }
        if (is_q) {
            return;
        }
        device ushort *at = cache_k + ((ulong)h * args.ctx + slot) * BLOOMERY_INK2_HEAD + lane;
        for (uint i = 0u; i < 4u; ++i) {
            at[32u * i] = qk_clip_half(x[i] * f, faults, args.layer);
        }
        return;
    }

    if (gw < heads + 2u * kv) {
        const uint h = gw - heads - kv;
        const uint base = h * BLOOMERY_INK2_HEAD;
        float x[4];
        float acc = 0.f;
        for (uint i = 0u; i < 4u; ++i) {
            x[i] = v[base + lane + 32u * i];
            acc = fma(x[i], x[i], acc);
        }
        const float sq = simd_sum(acc);
        const float scale = rms_scale(sq, BLOOMERY_INK2_HEAD, args.v_eps);
        if (!rms_usable(sq, scale)) {
            if (lane == 0u) {
                raise(faults, args.layer, BLOOMERY_FAULT_SITE_CACHE_VALUE);
            }
            for (uint i = 0u; i < 4u; ++i) {
                v[base + lane + 32u * i] = NAN;
            }
            return;
        }
        device ushort *at = cache_v + ((ulong)h * args.ctx + slot) * BLOOMERY_INK2_HEAD + lane;
        for (uint i = 0u; i < 4u; ++i) {
            x[i] = x[i] * scale;
            v[base + lane + 32u * i] = x[i];
            at[32u * i] = qk_clip_half(x[i], faults, args.layer);
        }
        return;
    }

    const uint h = gw - heads - 2u * kv;
    const uint base = h * BLOOMERY_INK2_REL;
    const bool mine = lane < BLOOMERY_INK2_REL;
    float x = mine ? gr[base + lane] * r[base + lane] : 0.0f;
    const float f = qk_clip_factor(sqrt(simd_sum(mine ? x * x : 0.0f)), args.r_thresh);
    if (!isfinite(f)) {
        if (lane == 0u) {
            raise(faults, args.layer, BLOOMERY_FAULT_SITE_CACHE_VALUE);
        }
    }
    if (mine) {
        r[base + lane] = isfinite(f) ? x * f * tau : NAN;
    }
}

// ---------------------------------------------------------------------------
// The rel-bias attention over rows.

#define RP_GQA (BLOOMERY_INK2_Q / BLOOMERY_INK2_KV)
#define RP_HEAD BLOOMERY_INK2_HEAD
#define RP_REL BLOOMERY_INK2_REL
#define RP_KEYS 16u
#define RP_SGS 4u
#define RP_PAD 8u
#define RP_STRIDE (RP_HEAD + RP_PAD)
#define RP_TILE_HALVES (RP_KEYS * RP_STRIDE)
#define RP_ROWS 8u
#define RP_BLOCK_ROWS (RP_SGS * RP_ROWS)
#define RP_DCHUNKS (RP_HEAD / 8u)
#define RP_ROW_WORDS (RP_HEAD / 8u)
#define RP_LOADS (RP_KEYS * RP_ROW_WORDS / (RP_SGS * 32u))
#define RP_LANE_DIMS (RP_HEAD / 32u)
// The bias tile's distance span: a block's rows sit at most
// ceil(32 / GQA) positions apart, and a key tile is 16 keys wide.
#define RP_DSPAN 32u

static_assert(RP_GQA * BLOOMERY_INK2_KV == BLOOMERY_INK2_Q, "the group covers the query heads");
static_assert(RP_BLOCK_ROWS * RP_STRIDE <= 2u * RP_TILE_HALVES,
              "the query stage fits in the K and V tiles it shares memory with");
static_assert(RP_LOADS * RP_SGS * 32u == RP_KEYS * RP_ROW_WORDS,
              "the threadgroup's loads cover a tile exactly");
static_assert((RP_BLOCK_ROWS + RP_GQA - 1u) / RP_GQA + RP_KEYS <= RP_DSPAN,
              "a block's distances over one key tile fit the bias tile");
static_assert(RP_LANE_DIMS == 4u, "a lane holds four values of a head of 128");

struct RelRowsArgs {
    uint rows;        // the chunk's rows
    uint from;        // the chunk's first position
    uint plane;       // the planes' rows: a ring's own, a plane's the context
    uint window;      // the last window of a row's live keys attended, 0 for all
    uint layer;
    float scale;      // the score's scale, 1/sqrt(head)
    uint q_stride;    // f32 between rows of q and of r
    uint sink_stride; // f32 between rows of the sinks
    uint out_stride;  // f32 between rows of the output
    uint ctx;         // the positions the chunk's last row is bounded by
};

static_assert(sizeof(RelRowsArgs) == 40, "RelRowsArgs is ten words");

// The first key a row at position p attends.
inline uint rp_lo(constant RelRowsArgs &args, uint p) {
    const uint live = p + 1u;
    return args.window != 0u && live > args.window ? live - args.window : 0u;
}

[[max_total_threads_per_threadgroup(RP_SGS * 32u)]]
kernel void rel_prefill_f16(device float const *q [[buffer(0)]],
                            device float const *r [[buffer(1)]],
                            device float const *bias [[buffer(2)]],
                            device float const *sinks [[buffer(3)]],
                            device ushort const *cache_k [[buffer(4)]],
                            device ushort const *cache_v [[buffer(5)]],
                            device float *out [[buffer(6)]],
                            constant RelRowsArgs &args [[buffer(7)]],
                            device atomic_uint *faults [[buffer(8)]],
                            uint tid [[thread_index_in_threadgroup]],
                            uint sg [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]],
                            uint3 group [[threadgroup_position_in_grid]],
                            uint3 tpg [[threadgroups_per_grid]])
{
    threadgroup half shared[2u * RP_TILE_HALVES];
    threadgroup float bias_tile[RP_REL * RP_DSPAN];
    threadgroup half *tile_k = shared;
    threadgroup half *tile_v = shared + RP_TILE_HALVES;

    const ushort qid = ushort(lane / 4u);
    const ushort fm = (qid & 4u) + ((lane / 2u) % 4u);
    const ushort fn = (qid & 2u) * 2u + (lane % 2u) * 2u;

    // The heaviest blocks (the last positions) are the first threadgroups.
    const uint kv = group.y;
    const uint total = args.rows * RP_GQA;
    const uint n0 = (tpg.x - 1u - group.x) * RP_BLOCK_ROWS;
    const uint first_row = n0 / RP_GQA;
    const uint last_row = (min(n0 + RP_BLOCK_ROWS, total) - 1u) / RP_GQA;
    const uint p_lo = args.from + first_row;
    const uint p_hi = args.from + last_row;
    // A chunk past the context: the key-count site, and NaN rows.
    const bool refused = p_hi >= args.ctx || p_hi >= args.plane && args.window == 0u;
    if (refused && tid == 0u) {
        raise(faults, args.layer, (BLOOMERY_FAULT_SITE_KEY_COUNT));
    }
    const uint sg_n0 = n0 + sg * RP_ROWS;
    const bool writes = sg_n0 < total;

    // This lane's row: its chunk row, position, query head, and window.
    const uint n = min(sg_n0 + fm, total - 1u);
    const uint row = n / RP_GQA;
    const uint head = kv * RP_GQA + n % RP_GQA;
    const uint p = args.from + row;
    const uint row_lo = rp_lo(args, p);
    const uint first_tile = rp_lo(args, p_lo) / RP_KEYS;

    // The SIMD group's eight query rows, rounded to half into the stage.
    bool overflow = false;
    uint bad_rows = 0u;
    for (uint rr = 0u; rr < RP_ROWS; ++rr) {
        const uint nr = min(sg_n0 + rr, total - 1u);
        const uint qrow = nr / RP_GQA;
        const uint qh = kv * RP_GQA + nr % RP_GQA;
        device float const *src =
            q + (ulong)qrow * args.q_stride + qh * RP_HEAD + lane * RP_LANE_DIMS;
        threadgroup half *dst = shared + (sg * RP_ROWS + rr) * RP_STRIDE + lane * RP_LANE_DIMS;
        bool bad = false;
        for (uint d = 0u; d < RP_LANE_DIMS; ++d) {
            const float y = refused ? 0.0f : src[d];
            const half hv = half(y);
            if (!isfinite(float(hv))) {
                bad = true;
                overflow = overflow || isfinite(y);
            }
            dst[d] = hv;
        }
        bad_rows |= uint(simd_any(bad)) << rr;
    }
    const bool unusable = refused || ((bad_rows >> fm) & 1u) != 0u;
    overflow = simd_any(overflow);
    if (overflow && lane == 0u && writes) {
        raise(faults, args.layer, BLOOMERY_FAULT_SITE_F32_PRODUCT);
    }
    // The lane's relative row, in registers.
    float rv[RP_REL];
    for (uint d = 0u; d < RP_REL; ++d) {
        rv[d] = r[(ulong)row * args.q_stride + head * RP_REL + d];
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_half8x8 qf[RP_DCHUNKS];
    #pragma clang loop unroll(full)
    for (uint c = 0u; c < RP_DCHUNKS; ++c) {
        simdgroup_load(qf[c], shared + sg * RP_ROWS * RP_STRIDE + c * 8u, RP_STRIDE);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_float8x8 acc[RP_DCHUNKS];
    #pragma clang loop unroll(full)
    for (uint c = 0u; c < RP_DCHUNKS; ++c) {
        acc[c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    float running_m = -INFINITY;
    float running_l = 0.0f;

    const uint block_keys = refused ? 0u : p_hi + 1u;
    const uint tiles = (block_keys + RP_KEYS - 1u) / RP_KEYS;
    for (uint t = first_tile; t < tiles; ++t) {
        const uint k0 = t * RP_KEYS;
        for (uint i = 0u; i < RP_LOADS; ++i) {
            const uint e = tid + i * (RP_SGS * 32u);
            const uint kr = e / RP_ROW_WORDS;
            const uint c = e % RP_ROW_WORDS;
            const uint key = k0 + kr;
            uint4 wk = uint4(0u);
            uint4 wv = uint4(0u);
            if (key < block_keys) {
                const ulong at = ((ulong)kv * args.plane + key % args.plane) * RP_ROW_WORDS + c;
                wk = ((device const uint4 *)cache_k)[at];
                wv = ((device const uint4 *)cache_v)[at];
            }
            *((threadgroup uint4 *)(tile_k + kr * RP_STRIDE + c * 8u)) = wk;
            *((threadgroup uint4 *)(tile_v + kr * RP_STRIDE + c * 8u)) = wv;
        }
        // The bias tile: column j holds distance p_lo - k0 - 15 + j.
        const int d_base = int(p_lo) - int(k0) - int(RP_KEYS - 1u);
        for (uint e = tid; e < RP_REL * RP_DSPAN; e += RP_SGS * 32u) {
            const uint d = e / RP_DSPAN;
            const int dist = d_base + int(e % RP_DSPAN);
            bias_tile[e] = dist >= 0 && dist < int(BLOOMERY_INK2_EXTENT)
                               ? bias[d * BLOOMERY_INK2_EXTENT + uint(dist)]
                               : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 s0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_float8x8 s1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        #pragma clang loop unroll(full)
        for (uint c = 0u; c < RP_DCHUNKS; ++c) {
            simdgroup_half8x8 k0f;
            simdgroup_half8x8 k1f;
            simdgroup_load(k0f, tile_k + c * 8u, RP_STRIDE, ulong2(0, 0), true);
            simdgroup_load(k1f, tile_k + 8u * RP_STRIDE + c * 8u, RP_STRIDE, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(s0, qf[c], k0f, s0);
            simdgroup_multiply_accumulate(s1, qf[c], k1f, s1);
        }

        // This lane's four scores: keys fn, fn + 1 of each fragment, each
        // the scaled product then the bias dot ascending d.
        float sv[4];
        bool live[4];
        float tile_max = -INFINITY;
        for (uint j = 0u; j < 4u; ++j) {
            const float raw = j < 2u ? s0.thread_elements()[j] : s1.thread_elements()[j - 2u];
            const uint key = k0 + (j / 2u) * 8u + fn + (j % 2u);
            live[j] = key <= p && key >= row_lo;
            const int col = int(p) - int(key) - d_base;
            float b = 0.0f;
            if (live[j]) {
                b = fma(rv[0], bias_tile[uint(col)], 0.0f);
                for (uint d = 1u; d < RP_REL; ++d) {
                    b = fma(rv[d], bias_tile[d * RP_DSPAN + uint(col)], b);
                }
            }
            sv[j] = raw * args.scale + b;
            if (live[j]) {
                tile_max = max(tile_max, sv[j]);
            }
        }
        tile_max = max(tile_max, simd_shuffle_xor(tile_max, ushort(1)));
        tile_max = max(tile_max, simd_shuffle_xor(tile_max, ushort(8)));
        const float new_m = max(running_m, tile_max);
        const float old_m = running_m;
        const float factor = old_m > -INFINITY ? exp(old_m - new_m) : 0.0f;
        float pw[4];
        float weights = 0.0f;
        for (uint j = 0u; j < 4u; ++j) {
            pw[j] = live[j] ? exp(sv[j] - new_m) : 0.0f;
            weights += pw[j];
        }
        running_l = running_l * factor + weights;
        running_m = new_m;
        #pragma clang loop unroll(full)
        for (uint c = 0u; c < RP_DCHUNKS; ++c) {
            acc[c].thread_elements()[0] = acc[c].thread_elements()[0] * factor;
            acc[c].thread_elements()[1] *= factor;
        }
        simdgroup_half8x8 p0;
        simdgroup_half8x8 p1;
        p0.thread_elements()[0] = half(pw[0]);
        p0.thread_elements()[1] = half(pw[1]);
        p1.thread_elements()[0] = half(pw[2]);
        p1.thread_elements()[1] = half(pw[3]);
        #pragma clang loop unroll(full)
        for (uint c = 0u; c < RP_DCHUNKS; ++c) {
            simdgroup_half8x8 v0f;
            simdgroup_half8x8 v1f;
            simdgroup_load(v0f, tile_v + c * 8u, RP_STRIDE);
            simdgroup_load(v1f, tile_v + 8u * RP_STRIDE + c * 8u, RP_STRIDE);
            simdgroup_multiply_accumulate(acc[c], p0, v0f, acc[c]);
            simdgroup_multiply_accumulate(acc[c], p1, v1f, acc[c]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // The row's weight sum over its four lanes, then the sink as one more
    // logit in the denominator and nothing in the value (the step's fold).
    running_l += simd_shuffle_xor(running_l, ushort(1));
    running_l += simd_shuffle_xor(running_l, ushort(8));
    if (sg_n0 + fm >= total) {
        return;
    }
    const float sink = refused ? 0.0f : sinks[(ulong)row * args.sink_stride + head];
    const bool bad_sink = isnan(sink) || sink == INFINITY;
    if (bad_sink && !refused && (lane % 2u) == 0u && (lane & 8u) == 0u) {
        raise(faults, args.layer, BLOOMERY_FAULT_SITE_F32_PRODUCT);
    }
    const float m = max(running_m, sink);
    const float rescale = running_m > -INFINITY ? exp(running_m - m) : 0.0f;
    const float l = running_l * rescale + (sink > -INFINITY ? exp(sink - m) : 0.0f);
    const bool whole = !unusable && !bad_sink && l > 0.0f && isfinite(m);
    #pragma clang loop unroll(full)
    for (uint c = 0u; c < RP_DCHUNKS; ++c) {
        for (uint e = 0u; e < 2u; ++e) {
            const uint dim = c * 8u + fn + e;
            const ulong at = (ulong)row * args.out_stride + head * RP_HEAD + dim;
            out[at] = whole ? acc[c].thread_elements()[e] * rescale / l : NAN;
        }
    }
}

// ---------------------------------------------------------------------------
// The biased sigmoid router over rows.

kernel void router_sigmoid_rows(device const float *rows [[buffer(0)]],
                                device const float *x [[buffer(1)]],
                                device const float *bias [[buffer(2)]],
                                device float *probs [[buffer(3)]],
                                device uint *ids [[buffer(4)]],
                                device float *weights [[buffer(5)]],
                                constant SigmoidArgs &args [[buffer(6)]],
                                device atomic_uint *faults [[buffer(7)]],
                                uint t [[threadgroup_position_in_grid]],
                                uint tid [[thread_position_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]],
                                uint sg [[simdgroup_index_in_threadgroup]])
{
    threadgroup float tg_score[SI_EXPERTS];
    threadgroup ulong tg_keys[SI_EXPERTS];
    threadgroup uint tg_bad[SI_SG];
    device const float *xr = x + (ulong)t * SI_K;
    for (uint e = sg; e < SI_EXPERTS; e += SI_SG) {
        device const float *row = rows + (ulong)e * (ulong)SI_K;
        float acc = 0.f;
        for (uint i = 0u; i < SI_FLOAT4S; ++i) {
            const float4 w = router_load4(row, lane + SI_LANES * i);
            const float4 xv = router_load4(xr, lane + SI_LANES * i);
            acc += w.x * xv.x;
            acc += w.y * xv.y;
            acc += w.z * xv.z;
            acc += w.w * xv.w;
        }
        const float score = 1.0f / (exp(-simd_sum(acc)) + 1.0f);
        if (lane == 0u) {
            tg_score[e] = score;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint mine_bad = 0u;
    for (uint e = tid; e < SI_EXPERTS; e += SI_THREADS) {
        const float score = tg_score[e];
        probs[(ulong)t * SI_EXPERTS + e] = score;
        const float sel = score + bias[e];
        mine_bad |= (isfinite(score) && isfinite(sel)) ? 0u : 1u;
        tg_keys[e] = argmax_key(sel, e);
    }
    mine_bad = simd_max(mine_bad);
    if (lane == 0u) {
        tg_bad[sg] = mine_bad;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg != 0u) {
        return;
    }
    device uint *row_ids = ids + (ulong)t * args.top_k;
    device float *row_w = weights + (ulong)t * args.top_k;
    uint bad = tg_bad[0];
    for (uint g = 1u; g < SI_SG; ++g) {
        bad = max(bad, tg_bad[g]);
    }
    if (bad != 0u) {
        if (lane == 0u) {
            raise(faults, args.layer, BLOOMERY_FAULT_SITE_ROUTER);
            for (uint s = 0u; s < args.top_k; ++s) {
                row_ids[s] = s;
                row_w[s] = NAN;
            }
        }
        return;
    }
    ulong held[SI_KEYS];
    for (uint j = 0u; j < SI_KEYS; ++j) {
        held[j] = tg_keys[lane + SI_LANES * j];
    }
    float chosen[SI_TOPK];
    for (uint s = 0u; s < args.top_k; ++s) {
        ulong local = 0u;
        for (uint j = 0u; j < SI_KEYS; ++j) {
            local = max(local, held[j]);
        }
        const uint hi = simd_max(uint(local >> 32));
        const uint lo = simd_max(uint(local >> 32u) == hi ? uint(local) : 0u);
        const ulong chosen_key = ((ulong)hi << 32) | (ulong)lo;
        for (uint j = 0u; j < SI_KEYS; ++j) {
            held[j] = held[j] == chosen_key ? 0u : held[j];
        }
        if (lane == 0u) {
            const uint winner = ~uint(chosen_key);
            row_ids[s] = winner;
            chosen[s] = tg_score[winner];
        }
    }
    if (lane == 0u) {
        float sum = 0.f;
        for (uint s = 0u; s < args.top_k; ++s) {
            sum += chosen[s];
        }
        const float divisor = sum + 1e-20f;
        for (uint s = 0u; s < args.top_k; ++s) {
            row_w[s] = chosen[s] / divisor * args.scale;
        }
    }
}

// ---------------------------------------------------------------------------
// The slots' sum, stored.

kernel void combine_rows_store(device const float *y [[buffer(0)]],
                               device const uint *inv [[buffer(1)]],
                               device float *x [[buffer(2)]],
                               constant CombineArgs &args [[buffer(3)]],
                               uint row [[threadgroup_position_in_grid]],
                               uint tid [[thread_position_in_threadgroup]],
                               uint threads [[threads_per_threadgroup]])
{
    device float4 *xr = (device float4 *)(x + (ulong)row * args.x_stride);
    for (uint f4 = tid; f4 < args.cols / 4u; f4 += threads) {
        float4 acc = float4(0.f);
        for (uint s = 0u; s != args.per_row; ++s) {
            const ulong place = inv[(ulong)row * args.per_row + s];
            acc += *((device const float4 *)(y + place * args.y_stride) + f4);
        }
        xr[f4] = acc;
    }
}

// The route table of the family's picks: route_table's count, serial scan
// and placement over SI_EXPERTS groups, every slot a routed pick. A slot's
// id past the stack raises the expert-id site and is placed in group 0, a
// defined access; the pass is poisoned by the raise.
kernel void ink2_route_table(device const uint *ids [[buffer(0)]],
                             device uint *sorted [[buffer(1)]],
                             device uint *inv [[buffer(2)]],
                             device uint4 *tiles [[buffer(3)]],
                             constant RouteTableArgs &args [[buffer(4)]],
                             device atomic_uint *faults [[buffer(5)]],
                             uint tid [[thread_position_in_threadgroup]])
{
    threadgroup atomic_uint count[SI_EXPERTS];
    threadgroup uint start[SI_EXPERTS];
    threadgroup uint first_tile[SI_EXPERTS];

    for (uint g = tid; g < SI_EXPERTS; g += ROUTE_THREADS) {
        atomic_store_explicit(&count[g], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tid; s < args.slots; s += ROUTE_THREADS) {
        const uint id = ids[s];
        const bool ok = id < args.n_experts;
        if (!ok) {
            raise(faults, args.layer, (BLOOMERY_FAULT_SITE_EXPERT_ID));
        }
        atomic_fetch_add_explicit(&count[ok ? id : 0u], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        uint place = 0u;
        uint tile = 0u;
        for (uint g = 0u; g < SI_EXPERTS; ++g) {
            const uint n = atomic_load_explicit(&count[g], memory_order_relaxed);
            start[g] = place;
            first_tile[g] = tile;
            place += n;
            tile += (n + (ROUTE_TILE - 1u)) / ROUTE_TILE;
        }
        tiles[0] = uint4(tile, 0u, 0u, 0u);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint g = tid; g < SI_EXPERTS; g += ROUTE_THREADS) {
        const uint n = atomic_load_explicit(&count[g], memory_order_relaxed);
        for (uint j = 0u; j * ROUTE_TILE < n; ++j) {
            tiles[1u + first_tile[g] + j] =
                uint4(g, start[g] + j * ROUTE_TILE, min((uint)ROUTE_TILE, n - j * ROUTE_TILE), 0u);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint g = tid; g < SI_EXPERTS; g += ROUTE_THREADS) {
        atomic_store_explicit(&count[g], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tid; s < args.slots; s += ROUTE_THREADS) {
        const uint id = ids[s];
        const uint g = id < args.n_experts ? id : 0u;
        const uint place =
            atomic_fetch_add_explicit(&count[g], 1u, memory_order_relaxed) + start[g];
        sorted[place] = s;
        inv[s] = place;
    }
}
