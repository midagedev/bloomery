// The projection epilogues of an Inkling 2 rel-bias visit, and the f16
// append of its key and value: one launch a visit covers the query rows,
// the key rows, the value rows and the relative rows.
//
// The rows a token brings — BLOOMERY_INK2_Q query rows of
// BLOOMERY_INK2_HEAD values, BLOOMERY_INK2_KV key rows, as many value rows,
// and BLOOMERY_INK2_Q relative rows of BLOOMERY_INK2_REL values — are the
// reference's epilogue order, one SIMD group a row, eight rows a
// threadgroup (the visit's 2 · (Q + KV) rows are two threadgroups):
//
// - a query row takes its head's scale row, is clipped to an L2 norm of at
//   most `qk_thresh`, then carries the token's `tau`;
// - a key row takes no scale and clips at the same threshold; a value row
//   takes a gainless RMS norm at `v_eps` and no clip; both append f16 into
//   this visit's cache planes, head-major [KV][ctx][HEAD] u16;
// - a relative row takes its head's scale row, clips at `r_thresh`, then
//   carries `tau`. A windowed visit passes `tau` 1.0 and the multiply is
//   exact; a windowless visit passes the log-scaled position factor.
//
// The clip is the reference's: each value scaled, squared in f32, the
// squares summed in the SIMD group's tree (simd_sum, this backend's shared
// reduction order — the reference sums the warp's squares in f64, a named
// band term between the two at a 128-wide row), the sum rooted, the root
// biased by the fixed 1e-6, the factor threshold / (root + 1e-6) clamped to
// [0, 1], one multiply back. The value row's RMS is the norm's own scale
// (rms_scale of norm.metal) over its squares.
//
// No silent failure: a row whose sum of squares or scale is not usable
// (rms_usable, norm.metal) or whose clip factor is not finite raises the
// cache-value site and the row is NaN; a position at or past `ctx` raises
// the cache-position site, leaves that token's key and value rows NaN and
// appends nothing; a finite value whose half conversion is not finite
// (past the half range) raises the quant-column site, the appended word
// being the half infinity.
//
// This file comes after fault.metal (raise), norm.metal (rms_scale,
// rms_usable) and attention.metal (attn_f32_to_f16, the half conversion the
// cache rows share).

#pragma METAL fp contract(off)

// The launch's argument block, bound by value.
struct QkClipArgs {
    uint ctx;        // the cache planes' positions
    uint pos;        // this token's append position
    uint layer;
    float v_eps;     // the value rows' RMS epsilon
    float qk_thresh; // the query and key rows' L2 threshold
    float r_thresh;  // the relative rows' L2 threshold
    float tau;       // the windowless visits' log-scaled position factor, 1.0 windowed
};

static_assert(sizeof(QkClipArgs) == 28, "QkClipArgs is 28 bytes: the host packs seven words");

// The clip's factor: threshold / (root + 1e-6) clamped to [0, 1], or NaN
// for a root that is not finite — the row has no defined clip.
inline float qk_clip_factor(float root, float threshold) {
    return isfinite(root) ? clamp(threshold / (root + 1e-6f), 0.0f, 1.0f) : NAN;
}

// The half conversion of a finite value: the bits, and the quant-column
// raise when a finite value converted past the half range.
inline ushort qk_clip_half(float value, device atomic_uint *faults, uint layer) {
    const ushort bits = attn_f32_to_f16(value);
    if (!isfinite(float(as_type<half>(bits))) && isfinite(value)) {
        raise(faults, layer, BLOOMERY_FAULT_SITE_QUANT_COLUMN);
    }
    return bits;
}

// A query row (one SIMD group): the head's scale row, the clip, then the token's
// `tau`.
inline void qk_clip_query_row(device float *q,
                              device const float *gq,
                              uint h,
                              uint lane,
                              constant QkClipArgs &args,
                              device atomic_uint *faults)
{
    const uint base = h * BLOOMERY_INK2_HEAD;
    float x[4];
    float acc = 0.f;
    for (uint i = 0u; i < 4u; ++i) {
        x[i] = q[base + lane + 32u * i];
        x[i] *= gq[base + lane + 32u * i];
        acc = fma(x[i], x[i], acc);
    }
    const float f = qk_clip_factor(sqrt(simd_sum(acc)), args.qk_thresh);
    if (!isfinite(f)) {
        if (lane == 0u) {
            raise(faults, args.layer, BLOOMERY_FAULT_SITE_CACHE_VALUE);
        }
        for (uint i = 0u; i < 4u; ++i) {
            q[base + lane + 32u * i] = NAN;
        }
        return;
    }
    for (uint i = 0u; i < 4u; ++i) {
        q[base + lane + 32u * i] = x[i] * f * args.tau;
    }
}

// A key row (one SIMD group) from its four values `x` (lane + 32 i of the
// head's row): the clip, the write back and the f16 append.
inline void qk_clip_key_row(thread float *x,
                            device float *k,
                            device ushort *cache_k,
                            uint h,
                            uint lane,
                            constant QkClipArgs &args,
                            device atomic_uint *faults)
{
    const uint base = h * BLOOMERY_INK2_HEAD;
    if (args.pos >= args.ctx) {
        if (lane == 0u) {
            raise(faults, args.layer, BLOOMERY_FAULT_SITE_CACHE_POS);
        }
        for (uint i = 0u; i < 4u; ++i) {
            k[base + lane + 32u * i] = NAN;
        }
        return;
    }
    float acc = 0.f;
    for (uint i = 0u; i < 4u; ++i) {
        acc = fma(x[i], x[i], acc);
    }
    const float f = qk_clip_factor(sqrt(simd_sum(acc)), args.qk_thresh);
    if (!isfinite(f)) {
        if (lane == 0u) {
            raise(faults, args.layer, BLOOMERY_FAULT_SITE_CACHE_VALUE);
        }
        for (uint i = 0u; i < 4u; ++i) {
            k[base + lane + 32u * i] = NAN;
        }
        return;
    }
    for (uint i = 0u; i < 4u; ++i) {
        k[base + lane + 32u * i] = x[i] * f;
    }
    device ushort *at = cache_k + (h * args.ctx + args.pos) * BLOOMERY_INK2_HEAD + lane;
    for (uint i = 0u; i < 4u; ++i) {
        at[32u * i] = qk_clip_half(x[i] * f, faults, args.layer);
    }
}

// A value row (one SIMD group) from its four values `x`: a gainless RMS norm,
// no clip, then the append.
inline void qk_clip_value_row(thread float *x,
                              device float *v,
                              device ushort *cache_v,
                              uint h,
                              uint lane,
                              constant QkClipArgs &args,
                              device atomic_uint *faults)
{
    const uint base = h * BLOOMERY_INK2_HEAD;
    if (args.pos >= args.ctx) {
        if (lane == 0u) {
            raise(faults, args.layer, BLOOMERY_FAULT_SITE_CACHE_POS);
        }
        for (uint i = 0u; i < 4u; ++i) {
            v[base + lane + 32u * i] = NAN;
        }
        return;
    }
    float acc = 0.f;
    for (uint i = 0u; i < 4u; ++i) {
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
    device ushort *at = cache_v + (h * args.ctx + args.pos) * BLOOMERY_INK2_HEAD + lane;
    for (uint i = 0u; i < 4u; ++i) {
        x[i] *= scale;
        v[base + lane + 32u * i] = x[i];
        at[32u * i] = qk_clip_half(x[i], faults, args.layer);
    }
}

// A relative row (one SIMD group): a BLOOMERY_INK2_REL-wide row, its lanes'
// low lanes the row's own values.
inline void qk_clip_rel_row(device float *r,
                            device const float *gr,
                            uint h,
                            uint lane,
                            constant QkClipArgs &args,
                            device atomic_uint *faults)
{
    const uint base = h * BLOOMERY_INK2_REL;
    const bool mine = lane < BLOOMERY_INK2_REL;
    float x = mine ? r[base + lane] * gr[base + lane] : 0.0f;
    const float f = qk_clip_factor(sqrt(simd_sum(mine ? x * x : 0.0f)), args.r_thresh);
    if (!isfinite(f)) {
        if (lane == 0u) {
            raise(faults, args.layer, BLOOMERY_FAULT_SITE_CACHE_VALUE);
        }
    }
    if (mine) {
        r[base + lane] = isfinite(f) ? x * f * args.tau : NAN;
    }
}

kernel void qk_clip_append(device float *q [[buffer(0)]],
                           device float *k [[buffer(1)]],
                           device float *v [[buffer(2)]],
                           device float *r [[buffer(3)]],
                           device const float *gq [[buffer(4)]],
                           device const float *gr [[buffer(5)]],
                           device ushort *cache_k [[buffer(6)]],
                           device ushort *cache_v [[buffer(7)]],
                           constant QkClipArgs &args [[buffer(8)]],
                           device atomic_uint *faults [[buffer(9)]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]],
                           uint3 group [[threadgroup_position_in_grid]])
{
    // The launch's row: the query heads first, then the key heads, the
    // value heads, the relative heads; one SIMD group a row, eight a
    // threadgroup.
    const uint heads = BLOOMERY_INK2_Q;
    const uint kv = BLOOMERY_INK2_KV;
    const uint gw = group.x * 8u + sg;
    if (gw >= 2u * (heads + kv)) {
        return;
    }
    if (gw < heads) {
        qk_clip_query_row(q, gq, gw, lane, args, faults);
        return;
    }
    if (gw < heads + kv) {
        const uint h = gw - heads;
        float x[4];
        for (uint i = 0u; i < 4u; ++i) {
            x[i] = k[h * BLOOMERY_INK2_HEAD + lane + 32u * i];
        }
        qk_clip_key_row(x, k, cache_k, h, lane, args, faults);
        return;
    }
    if (gw < heads + 2u * kv) {
        const uint h = gw - heads - kv;
        float x[4];
        for (uint i = 0u; i < 4u; ++i) {
            x[i] = v[h * BLOOMERY_INK2_HEAD + lane + 32u * i];
        }
        qk_clip_value_row(x, v, cache_v, h, lane, args, faults);
        return;
    }
    qk_clip_rel_row(r, gr, gw - heads - 2u * kv, lane, args, faults);
}

// The same launch with the key and value short convs run in its tail: a key or
// value row's SIMD group is one head (128 channels, the head's dynamic taps
// all its own), so it reads the projection's raw row, the filter, the ring's
// three predecessors and the head's taps, runs the conv on each of its four
// values (conv_res_four, conv_res_four_dyn: conv_res's rule at one token), stores
// the raw input into the ring after its reads, and goes on to the clip or the
// norm of the conv's output as qk_clip_append does of the conv's row. The conv's
// row itself is never stored: the clip rewrote it in place.
//
// A conv output that is not finite raises the linear-conv site, as conv_res's
// does, and goes on into the clip or the norm as it is: a row with a value that
// is not finite has a sum of squares that is not, and both raise the
// cache-value site and write a NaN row whether the value was NaN or infinite,
// which is what conv_res's NaN gave them. The position past the plane is still
// raised after the conv ran (the unfolded pair's order).
struct QkClipConvArgs {
    QkClipArgs clip;
    uint ring_rows;  // the conv ring's rows
    uint ch;         // the ring's row width
    uint base_k;     // the k conv's first channel in the ring's rows
    uint base_v;     // the v conv's first channel
    uint heads;      // the dynamic taps' heads: 0 a static conv, else BLOOMERY_INK2_KV
    uint pos;        // this token's position in the sequence (the conv's, not the plane's)
    float scale;     // the conv's output scale
};

static_assert(sizeof(QkClipConvArgs) == 56, "QkClipConvArgs is 56 bytes: the host packs fourteen words");

// The conv of one key or value head's four values per lane: x receives the four
// outputs (NaN where one is not finite).
inline void qk_clip_conv_row(device const float *raw,
                             device const float *w,
                             device const float *dyn,
                             device float *ring,
                             uint base,
                             uint h,
                             uint lane,
                             constant QkClipConvArgs &a,
                             device atomic_uint *faults,
                             thread float *x)
{
    float cur[4];
    float4 tap[4];
    float p3[4];
    float p2[4];
    float p1[4];
    uint at[4];
    for (uint i = 0u; i < 4u; ++i) {
        const uint c = h * BLOOMERY_INK2_HEAD + lane + 32u * i;
        at[i] = base + c;
        cur[i] = raw[c];
        tap[i] = *((device const float4 *)w + c);
        p3[i] = conv_res_past(ring, a.ring_rows, a.ch, at[i], a.pos, 3u);
        p2[i] = conv_res_past(ring, a.ring_rows, a.ch, at[i], a.pos, 2u);
        p1[i] = conv_res_past(ring, a.ring_rows, a.ch, at[i], a.pos, 1u);
    }
    const bool dynamic = a.heads > 0u;
    const float4 dy = dynamic ? *((device const float4 *)dyn + h) : float4(0.0f);
    bool ok = true;
    for (uint i = 0u; i < 4u; ++i) {
        const float out = dynamic
            ? conv_res_four_dyn(tap[i], dy, p3[i], p2[i], p1[i], cur[i], a.scale)
            : conv_res_four(tap[i], p3[i], p2[i], p1[i], cur[i], a.scale);
        ok = ok && isfinite(out);
        x[i] = out;
        ring[(ulong)sc_slot(a.pos, a.ring_rows) * a.ch + at[i]] = cur[i];
    }
    if (!ok) {
        raise(faults, a.clip.layer, BLOOMERY_FAULT_SITE_LINEAR_CONV);
    }
}

kernel void qk_clip_conv_append(device float *q [[buffer(0)]],
                                device float *k [[buffer(1)]],
                                device float *v [[buffer(2)]],
                                device float *r [[buffer(3)]],
                                device const float *gq [[buffer(4)]],
                                device const float *gr [[buffer(5)]],
                                device ushort *cache_k [[buffer(6)]],
                                device ushort *cache_v [[buffer(7)]],
                                constant QkClipConvArgs &args [[buffer(8)]],
                                device atomic_uint *faults [[buffer(9)]],
                                device const float *k_raw [[buffer(10)]],
                                device const float *v_raw [[buffer(11)]],
                                device const float *wk [[buffer(12)]],
                                device const float *wv [[buffer(13)]],
                                device const float *dyn_k [[buffer(14)]],
                                device const float *dyn_v [[buffer(15)]],
                                device float *ring [[buffer(16)]],
                                uint lane [[thread_index_in_simdgroup]],
                                uint sg [[simdgroup_index_in_threadgroup]],
                                uint3 group [[threadgroup_position_in_grid]])
{
    const uint heads = BLOOMERY_INK2_Q;
    const uint kv = BLOOMERY_INK2_KV;
    const uint gw = group.x * 8u + sg;
    if (gw >= 2u * (heads + kv)) {
        return;
    }
    if (gw < heads) {
        qk_clip_query_row(q, gq, gw, lane, args.clip, faults);
        return;
    }
    if (gw < heads + kv) {
        const uint h = gw - heads;
        float x[4];
        qk_clip_conv_row(k_raw, wk, dyn_k, ring, args.base_k, h, lane, args, faults, x);
        qk_clip_key_row(x, k, cache_k, h, lane, args.clip, faults);
        return;
    }
    if (gw < heads + 2u * kv) {
        const uint h = gw - heads - kv;
        float x[4];
        qk_clip_conv_row(v_raw, wv, dyn_v, ring, args.base_v, h, lane, args, faults, x);
        qk_clip_value_row(x, v, cache_v, h, lane, args.clip, faults);
        return;
    }
    qk_clip_rel_row(r, gr, gw - heads - 2u * kv, lane, args.clip, faults);
}
