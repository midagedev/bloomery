// The router, one dispatch. Geometry (ii): one row per SIMD group, split-K 2,
// eight float4s per lane. 257 rows of K 2048 f32. Row 256 reads the second
// segment base; the two bases are not joined. 514 SIMD groups, so 129
// threadgroups of 128 threads, and the last threadgroup has two SIMD groups
// with no row.
//
// Each SIMD group reduces the sum of squares of the whole input itself. The
// gain multiplies each value as it is loaded, and the scale multiplies the
// finished partial once: the dot is linear in the input. The squares are of
// the raw input. The last threadgroup, in route_last, adds split 0 then
// split 1, selects, writes the weights and resets the ticket.

#pragma METAL fp contract(off)

#define ROUTER_ROWS BLOOMERY_ROUTER_ROWS
#define ROUTER_EXPERTS BLOOMERY_ROUTER_EXPERTS
#define ROUTER_K BLOOMERY_ROUTER_K
#define ROUTER_SPLIT BLOOMERY_ROUTER_SPLIT
#define ROUTER_SG BLOOMERY_ROUTER_SG
#define ROUTER_FLOAT4S BLOOMERY_ROUTER_FLOAT4S
#define ROUTER_TOPK_MAX BLOOMERY_ROUTER_TOPK_MAX
#define ROUTER_LANES BLOOMERY_ROUTER_LANES
#define ROUTER_WORK (ROUTER_ROWS * ROUTER_SPLIT)

static_assert(ROUTER_ROWS == ROUTER_EXPERTS + 1, "the shared expert is the last row");
static_assert(ROUTER_WORK == ROUTER_ROWS * ROUTER_SPLIT, "the work is the rows times the splits");
static_assert(ROUTER_LANES * ROUTER_FLOAT4S * 4 == ROUTER_K / ROUTER_SPLIT,
              "the float4s cover one split");
static_assert(ROUTER_LANES * ROUTER_FLOAT4S * ROUTER_SPLIT * 4 == ROUTER_K,
              "the splits' float4s cover K");

// The argument block, bound by value. The host packs three words. The grid's
// threadgroup count is `[[threadgroups_per_grid]]`, not a word of this block.
struct RouterArgs {
    uint top_k;
    uint layer;
    float eps;
};

static_assert(sizeof(RouterArgs) == 12, "RouterArgs is 12 bytes: the host packs three words");

inline float4 router_load4(device const float *p, uint f4)
{
    return *((device const float4 *)(p + f4 * 4u));
}

// The last threadgroup's whole epilogue: the split add, the selection, the
// weights and the reset. One thread calls it, after ticket_acquire.
static void route_last(device coherent(device) float *partials,
                       device atomic_uint *ticket,
                       device float *logits,
                       device uint *ids,
                       device float *weights,
                       device atomic_uint *faults,
                       constant RouterArgs &args)
{
    ticket_acquire();
    bool bad = false;
    for (uint row = 0u; row < ROUTER_ROWS; ++row) {
        float p0 = ticket_load(partials, row * ROUTER_SPLIT);
        float p1 = ticket_load(partials, row * ROUTER_SPLIT + 1u);
        float l = p0 + p1;
        logits[row] = l;
        if (!isfinite(l)) {
            bad = true;
        }
    }
    if (bad) {
        raise(faults, args.layer, BLOOMERY_FAULT_SITE_ROUTER);
        for (uint s = 0u; s < args.top_k; ++s) {
            ids[s] = s;
            weights[s] = NAN;
        }
        ids[args.top_k] = ROUTER_EXPERTS;
        weights[args.top_k] = NAN;
    } else {
        bool taken[ROUTER_EXPERTS];
        for (uint i = 0u; i < ROUTER_EXPERTS; ++i) {
            taken[i] = false;
        }
        float chosen_v[ROUTER_TOPK_MAX];
        for (uint s = 0u; s < args.top_k; ++s) {
            bool any = false;
            float best_v = 0.f;
            uint best_i = 0u;
            for (uint i = 0u; i < ROUTER_EXPERTS; ++i) {
                if (taken[i]) {
                    continue;
                }
                float v = logits[i];
                if (!any || v > best_v) {
                    any = true;
                    best_v = v;
                    best_i = i;
                }
            }
            taken[best_i] = true;
            chosen_v[s] = best_v;
            ids[s] = best_i;
        }
        float l_max = chosen_v[0];
        float term[ROUTER_TOPK_MAX];
        float sum_exp = 0.f;
        for (uint s = 0u; s < args.top_k; ++s) {
            term[s] = exp(chosen_v[s] - l_max);
            sum_exp += term[s];
        }
        for (uint s = 0u; s < args.top_k; ++s) {
            weights[s] = term[s] / sum_exp;
        }
        ids[args.top_k] = ROUTER_EXPERTS;
        weights[args.top_k] = 1.f / (1.f + exp(-logits[ROUTER_EXPERTS]));
    }
    ticket_reset(ticket);
}

kernel void router35(device const float *routed [[buffer(0)]],
                     device const float *shared [[buffer(1)]],
                     device const float *x [[buffer(2)]],
                     device const float *gain [[buffer(3)]],
                     device coherent(device) float *partials [[buffer(4)]],
                     device atomic_uint *ticket [[buffer(5)]],
                     device float *logits [[buffer(6)]],
                     device uint *ids [[buffer(7)]],
                     device float *weights [[buffer(8)]],
                     constant RouterArgs &args [[buffer(9)]],
                     device atomic_uint *faults [[buffer(10)]],
                     uint tg [[threadgroup_position_in_grid]],
                     uint tid [[thread_position_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]],
                     uint sg_in_tg [[simdgroup_index_in_threadgroup]],
                     uint groups [[threadgroups_per_grid]])
{
    threadgroup uint last_flag = 0u;

    const uint sg = tg * ROUTER_SG + sg_in_tg;
    if (sg < ROUTER_WORK) {
        const uint row = sg / ROUTER_SPLIT;
        const uint split = sg % ROUTER_SPLIT;
        float sum_sq = 0.f;
        for (uint i = 0u; i < 16u; ++i) {
            float4 xv = router_load4(x, lane + ROUTER_LANES * i);
            sum_sq += xv.x * xv.x + xv.y * xv.y + xv.z * xv.z + xv.w * xv.w;
        }
        sum_sq = simd_sum(sum_sq);
        float scale = rms_scale(sum_sq, ROUTER_K, args.eps);
        if (!rms_usable(sum_sq, scale)) {
            if (lane == 0u) {
                raise(faults, args.layer, BLOOMERY_FAULT_SITE_NORM_QUANT);
                ticket_write(partials, sg, NAN);
            }
        } else {
            device const float *row_base = row < ROUTER_EXPERTS
                ? routed + (ulong)row * (ulong)ROUTER_K
                : shared;
            const uint base4 = split * (ROUTER_K / ROUTER_SPLIT / 4u);
            float acc = 0.f;
#pragma unroll(8)
            for (uint i = 0u; i < ROUTER_FLOAT4S; ++i) {
                const uint f4 = base4 + lane + ROUTER_LANES * i;
                float4 xv = router_load4(x, f4);
                float4 gv = router_load4(gain, f4);
                float4 h = xv * gv;
                float4 w = router_load4(row_base, f4);
                acc += w.x * h.x;
                acc += w.y * h.y;
                acc += w.z * h.z;
                acc += w.w * h.w;
            }
            acc = simd_sum(acc) * scale;
            if (lane == 0u) {
                ticket_write(partials, sg, acc);
            }
        }
    }

    ticket_is_last(ticket, groups, tid, last_flag);
    if (last_flag == 1u && tid == 0u) {
        route_last(partials, ticket, logits, ids, weights, faults, args);
    }
}
