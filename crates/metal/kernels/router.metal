// The router, one dispatch. Geometry: one row per SIMD group, no split, so
// 257 SIMD groups of one row of K 2048 f32, sixteen float4s per lane. Eight
// SIMD groups a threadgroup make 256 threads and 33 threadgroups; the last
// threadgroup has seven SIMD groups with no row. Row 256 reads the second
// segment base; the two bases are not joined.
//
// The norm is the threadgroup's, not the row's: every lane loads its share of
// the input and the gain (two float4s at 256 lanes), stores the gained value
// in threadgroup memory for the row loop, and sums its squares; one thread
// adds the SIMD groups' shares and takes rsqrt. The scale multiplies each
// finished row once: the dot is linear in the input.
//
// A row's dot: the lane loads its sixteen float4s of weights and of the
// gained input in i order, accumulating the four products of each into one
// accumulator (64 products, 63 adds), the SIMD group sums the lanes, and the
// scale multiplies once. The last threadgroup, in route_last, reads the
// rows across its threads, and its first SIMD group selects by rounds of its
// largest packed key; lane 0 writes the weights, and the ticket resets.

#pragma METAL fp contract(off)

#define ROUTER_ROWS BLOOMERY_ROUTER_ROWS
#define ROUTER_EXPERTS BLOOMERY_ROUTER_EXPERTS
#define ROUTER_K BLOOMERY_ROUTER_K
#define ROUTER_SPLIT BLOOMERY_ROUTER_SPLIT
#define ROUTER_SG BLOOMERY_ROUTER_SG
#define ROUTER_FLOAT4S BLOOMERY_ROUTER_FLOAT4S
#define ROUTER_TOPK_MAX BLOOMERY_ROUTER_TOPK_MAX
#define ROUTER_LANES BLOOMERY_ROUTER_LANES
#define ROUTER_THREADS (ROUTER_LANES * ROUTER_SG)
#define ROUTER_KEYS (ROUTER_EXPERTS / ROUTER_LANES)
#define ROUTER_WORK (ROUTER_ROWS * ROUTER_SPLIT)

static_assert(ROUTER_ROWS == ROUTER_EXPERTS + 1, "the shared expert is the last row");
static_assert(ROUTER_WORK == ROUTER_ROWS, "the work is the rows: no split");
static_assert(ROUTER_LANES * ROUTER_FLOAT4S * 4 == ROUTER_K / ROUTER_SPLIT,
              "the float4s cover one split");
static_assert(ROUTER_LANES * ROUTER_FLOAT4S * ROUTER_SPLIT * 4 == ROUTER_K,
              "the splits' float4s cover K");
static_assert(ROUTER_K / 4u % ROUTER_THREADS == 0u,
              "the norm's lanes cover the input exactly");
static_assert(ROUTER_EXPERTS % ROUTER_LANES == 0u,
              "the selection's lanes take whole rows of keys");

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

// The selection, from the logits and packed keys a threadgroup has published
// in threadgroup memory: the first SIMD group alone, after the barrier that
// published them. The group reduces the "bad" flags (a logit that is not
// finite raises the router site and leaves NaN weights over the ids 0.. in
// order); else its lanes take the keys their rows left, eight each at 32
// lanes over the routed experts, and the selection runs in rounds of the
// group's largest packed key reduced by two SIMD maximums, with no further
// barrier. The weights stay on lane 0, serial, so their bits stay put. The
// ticket, when the caller holds one, resets after the weights. The decode's
// last threadgroup and the rows kernel both end here: the ids and weights
// pointers are the row's.
static void route_pick(device uint *ids,
                       device float *weights,
                       device atomic_uint *ticket,
                       device atomic_uint *faults,
                       uint top_k,
                       uint layer,
                       uint lane,
                       threadgroup float *tg_logits,
                       threadgroup ulong *tg_keys,
                       threadgroup uint *tg_bad)
{
    uint bad = tg_bad[0];
    for (uint g = 1u; g < ROUTER_SG; ++g) {
        bad = max(bad, tg_bad[g]);
    }
    if (bad != 0u) {
        if (lane == 0u) {
            raise(faults, layer, BLOOMERY_FAULT_SITE_ROUTER);
            for (uint s = 0u; s < top_k; ++s) {
                ids[s] = s;
                weights[s] = NAN;
            }
            ids[top_k] = ROUTER_EXPERTS;
            weights[top_k] = NAN;
            if (ticket != nullptr) {
                ticket_reset(ticket);
            }
        }
        return;
    }
    // The group's lanes take the keys their rows left, eight each at 32
    // lanes over the routed experts.
    ulong held[ROUTER_KEYS];
    for (uint j = 0u; j < ROUTER_KEYS; ++j) {
        held[j] = tg_keys[lane + ROUTER_LANES * j];
    }
    // One round per selected expert: the group's largest key names this
    // round's selection, and the round's winner is keyed out of the next.
    // The key's order is the old scan's: the larger value first, at an equal
    // value the smaller id.
    float chosen_v[ROUTER_TOPK_MAX];
    for (uint s = 0u; s < top_k; ++s) {
        ulong local = 0u;
        for (uint j = 0u; j < ROUTER_KEYS; ++j) {
            local = max(local, held[j]);
        }
        // The group's key, reduced as ticket_max_key reduces a group's: the
        // high half's maximum, then the low halves' maximum of the lanes that
        // hold it.
        uint o = uint(local >> 32);
        uint hi = simd_max(o);
        uint lo = simd_max(o == hi ? uint(local) : 0u);
        const ulong chosen_key = ((ulong)hi << 32) | (ulong)lo;
        for (uint j = 0u; j < ROUTER_KEYS; ++j) {
            if (held[j] == chosen_key) {
                held[j] = 0u;
            }
        }
        if (lane == 0u) {
            uint best_i = ~uint(chosen_key);
            ids[s] = best_i;
            chosen_v[s] = tg_logits[best_i];
        }
    }
    // The softmax and the shared expert's sigmoid stay on one thread, in the
    // order the serial epilogue once used, so the weights' bits do not move.
    if (lane == 0u) {
        float l_max = chosen_v[0];
        float term[ROUTER_TOPK_MAX];
        float sum_exp = 0.f;
        for (uint s = 0u; s < top_k; ++s) {
            term[s] = exp(chosen_v[s] - l_max);
            sum_exp += term[s];
        }
        for (uint s = 0u; s < top_k; ++s) {
            weights[s] = term[s] / sum_exp;
        }
        ids[top_k] = ROUTER_EXPERTS;
        weights[top_k] = 1.f / (1.f + exp(-tg_logits[ROUTER_EXPERTS]));
        if (ticket != nullptr) {
            ticket_reset(ticket);
        }
    }
}

// The last threadgroup's whole epilogue, after ticket_acquire: every thread
// reads its one or two rows' partials (the whole threadgroup's loads at
// once), writes the logits and the packed keys to threadgroup memory, and one
// barrier publishes them; the first SIMD group then selects (route_pick).
static void route_last(device coherent(device) float *partials,
                       device atomic_uint *ticket,
                       device float *logits,
                       device uint *ids,
                       device float *weights,
                       device atomic_uint *faults,
                       constant RouterArgs &args,
                       uint tid,
                       uint lane,
                       uint sg_in_tg,
                       threadgroup float *tg_logits,
                       threadgroup ulong *tg_keys,
                       threadgroup uint *tg_bad)
{
    ticket_acquire();
    // Each thread reads its rows' partials, the values one lane wrote each, so
    // every logit is the bits it was. The keys are ticket.metal's argmax_key
    // of the same value.
    uint mine_bad = 0u;
    for (uint row = tid; row < ROUTER_ROWS; row += ROUTER_THREADS) {
        float l = ticket_load(partials, row);
        logits[row] = l;
        tg_logits[row] = l;
        mine_bad |= isfinite(l) ? 0u : 1u;
        // The shared expert's row is not a selection candidate; the old scan
        // read only the routed experts.
        if (row < ROUTER_EXPERTS) {
            tg_keys[row] = argmax_key(l, row);
        }
    }
    mine_bad = simd_max(mine_bad);
    if (lane == 0u) {
        tg_bad[sg_in_tg] = mine_bad;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_in_tg != 0u) {
        return;
    }
    route_pick(ids, weights, ticket, faults, args.top_k, args.layer, lane, tg_logits, tg_keys,
               tg_bad);
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
    threadgroup float4 tg_h[ROUTER_K / 4u];
    threadgroup float tg_logits[ROUTER_ROWS];
    threadgroup ulong tg_keys[ROUTER_EXPERTS];
    threadgroup uint tg_bad[ROUTER_SG];
    threadgroup float tg_sums[ROUTER_SG];
    threadgroup float tg_scale[1];

    // The norm, once per threadgroup: every lane loads its float4s of the
    // input and the gain, stores the gained value, and sums its squares; one
    // thread adds the SIMD groups' shares and takes rsqrt.
    float sum_sq = 0.f;
    for (uint i = 0u; i < ROUTER_K / 4u / ROUTER_THREADS; ++i) {
        const uint f4 = tid + ROUTER_THREADS * i;
        float4 xv = router_load4(x, f4);
        float4 gv = router_load4(gain, f4);
        tg_h[f4] = xv * gv;
        sum_sq += xv.x * xv.x + xv.y * xv.y + xv.z * xv.z + xv.w * xv.w;
    }
    sum_sq = simd_sum(sum_sq);
    if (lane == 0u) {
        tg_sums[sg_in_tg] = sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float total = 0.f;
        for (uint g = 0u; g < ROUTER_SG; ++g) {
            total += tg_sums[g];
        }
        tg_scale[0] = rms_scale(total, ROUTER_K, args.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Every thread reads the sums back and takes the same decision, so the
    // group's rows all write real partials or all write NaN.
    float total = 0.f;
    for (uint g = 0u; g < ROUTER_SG; ++g) {
        total += tg_sums[g];
    }
    const float scale = tg_scale[0];
    const uint sg = tg * ROUTER_SG + sg_in_tg;
    if (!rms_usable(total, scale)) {
        if (tid == 0u) {
            raise(faults, args.layer, BLOOMERY_FAULT_SITE_NORM_QUANT);
        }
        if (sg < ROUTER_ROWS && lane == 0u) {
            ticket_write(partials, sg, NAN);
        }
    } else if (sg < ROUTER_ROWS) {
        device const float *row_base = sg < ROUTER_EXPERTS
            ? routed + (ulong)sg * (ulong)ROUTER_K
            : shared;
        float acc = 0.f;
#pragma unroll(8)
        for (uint i = 0u; i < ROUTER_FLOAT4S; ++i) {
            const uint f4 = lane + ROUTER_LANES * i;
            float4 h = tg_h[f4];
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

    ticket_is_last(ticket, groups, tid, last_flag);
    if (last_flag == 1u) {
        route_last(partials, ticket, logits, ids, weights, faults, args,
                   tid, lane, sg_in_tg, tg_logits, tg_keys, tg_bad);
    }
}
