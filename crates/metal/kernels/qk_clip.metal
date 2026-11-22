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
// - a query row takes its per-channel scale, is clipped to an L2 norm of at
//   most `qk_thresh`, then carries the token's `tau`;
// - a key row takes no scale and clips at the same threshold; a value row
//   takes a gainless RMS norm at `v_eps` and no clip; both append f16 into
//   this visit's cache planes, head-major [KV][ctx][HEAD] u16;
// - a relative row takes its per-channel scale, clips at `r_thresh`, then
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
// (past the half range) raises the quant-column site and appends NaN.
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

    // The query heads, then the key heads. `is_q` is SIMD-group-uniform.
    if (gw < heads + kv) {
        const bool is_q = gw < heads;
        const uint h = is_q ? gw : gw - heads;
        const uint base = h * BLOOMERY_INK2_HEAD;
        device float *rowbuf = is_q ? q : k;
        if (!is_q && args.pos >= args.ctx) {
            if (lane == 0u) {
                raise(faults, args.layer, BLOOMERY_FAULT_SITE_CACHE_POS);
            }
            for (uint i = 0u; i < 4u; ++i) {
                rowbuf[base + lane + 32u * i] = NAN;
            }
            return;
        }
        float x[4];
        float acc = 0.f;
        for (uint i = 0u; i < 4u; ++i) {
            x[i] = rowbuf[base + lane + 32u * i];
            if (is_q) {
                x[i] *= gq[lane + 32u * i];
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
            rowbuf[base + lane + 32u * i] = is_q ? x[i] * f * args.tau : x[i] * f;
        }
        if (is_q) {
            return; // the query rows append nothing
        }
        device ushort *at = cache_k + (h * args.ctx + args.pos) * BLOOMERY_INK2_HEAD + lane;
        for (uint i = 0u; i < 4u; ++i) {
            at[32u * i] = qk_clip_half(x[i] * f, faults, args.layer);
        }
        return;
    }

    // The value heads: a gainless RMS norm, no clip, then the append.
    if (gw < heads + 2u * kv) {
        const uint h = gw - heads - kv;
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
        device ushort *at = cache_v + (h * args.ctx + args.pos) * BLOOMERY_INK2_HEAD + lane;
        for (uint i = 0u; i < 4u; ++i) {
            x[i] *= scale;
            v[base + lane + 32u * i] = x[i];
            at[32u * i] = qk_clip_half(x[i], faults, args.layer);
        }
        return;
    }

    // The relative heads: a BLOOMERY_INK2_REL-wide row, its lanes' low
    // lanes the row's own values.
    const uint h = gw - heads - 2u * kv;
    const uint base = h * BLOOMERY_INK2_REL;
    const bool mine = lane < BLOOMERY_INK2_REL;
    float x = mine ? r[base + lane] * gr[lane] : 0.0f;
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
