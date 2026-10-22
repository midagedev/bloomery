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

#define ROUTER_ROWS 257
#define ROUTER_EXPERTS 256
#define ROUTER_K 2048
#define ROUTER_SPLIT 2
#define ROUTER_SG 4
#define ROUTER_FLOAT4S 8
#define ROUTER_TOPK_MAX 8
#define ROUTER_LANES 32
#define ROUTER_WORK (ROUTER_ROWS * ROUTER_SPLIT)

static_assert(ROUTER_ROWS == ROUTER_EXPERTS + 1, "the shared expert is the last row");
static_assert(ROUTER_WORK == 514, "257 rows and two splits");
static_assert(ROUTER_LANES * ROUTER_FLOAT4S * 4 == ROUTER_K / ROUTER_SPLIT,
              "eight float4s cover one split");
static_assert(ROUTER_LANES * 16 * 4 == ROUTER_K, "sixteen float4s cover K");
static_assert(ROUTER_SG * ROUTER_LANES == 128, "a threadgroup is four SIMD groups");

// The argument block, bound by value. The host packs four words.
struct RouterArgs {
    uint top_k;
    uint layer;
    float eps;
    uint groups;
};

static_assert(sizeof(RouterArgs) == 16, "RouterArgs is 16 bytes: the host packs four words");

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
        uint router_chosen[ROUTER_TOPK_MAX];
        for (uint s = 0u; s < args.top_k; ++s) {
            bool any = false;
            float best_v = 0.f;
            uint best_i = 0u;
            for (uint i = 0u; i < ROUTER_EXPERTS; ++i) {
                if (taken[i]) {
                    continue;
                }
                float v = logits[i];
                bool router_better = v > best_v;
                if (!any || router_better) {
                    any = true;
                    best_v = v;
                    best_i = i;
                }
            }
            taken[best_i] = true;
            chosen_v[s] = best_v;
            router_chosen[s] = best_i;
        }
        float l_max = chosen_v[0];
        float term[ROUTER_TOPK_MAX];
        float sum_exp = 0.f;
        for (uint s = 0u; s < args.top_k; ++s) {
            term[s] = exp(chosen_v[s] - l_max);
            sum_exp += term[s];
        }
        for (uint s = 0u; s < args.top_k; ++s) {
            float router_weight = term[s] / sum_exp;
            weights[s] = router_weight;
            ids[s] = router_chosen[s];
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
                     uint sg_in_tg [[simdgroup_index_in_threadgroup]])
{
    threadgroup uint last_flag = 0u;

    // A launch the host accepted passes this. The bounds mutant flips the
    // comparison so every thread returns before a barrier.
    uint router_bounds_gate = 1u;
    if (router_bounds_gate == 0u) {
        return;
    }

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
            float router_acc = 0.f;
#pragma unroll(8)
            for (uint i = 0u; i < ROUTER_FLOAT4S; ++i) {
                const uint f4 = base4 + lane + ROUTER_LANES * i;
                float4 xv = router_load4(x, f4);
                float4 gv = router_load4(gain, f4);
                float4 h = xv * gv;
                float4 w = router_load4(row_base, f4);
                router_acc += w.x * h.x;
                router_acc += w.y * h.y;
                router_acc += w.z * h.z;
                router_acc += w.w * h.w;
            }
            router_acc = simd_sum(router_acc) * scale;
            if (lane == 0u) {
                ticket_write(partials, sg, router_acc);
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_device);
    if (tid == 0u) {
        uint old = ticket_arrive(ticket);
        last_flag = (old + 1u == args.groups) ? 1u : 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (last_flag == 1u && tid == 0u) {
        route_last(partials, ticket, logits, ids, weights, faults, args);
    }
}
