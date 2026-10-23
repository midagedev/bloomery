// The router, one dispatch. Geometry (ii): one row per SIMD group, split-K 2,
// eight float4s per lane. 257 rows of K 2048 f32. Row 256 reads the second
// segment base; the two bases are not joined. 514 SIMD groups, so 129
// threadgroups of 128 threads, and the last threadgroup has two SIMD groups
// with no row.
//
// Each SIMD group reduces the sum of squares of the whole input itself. The
// gain multiplies each value as it is loaded, and the scale multiplies the
// finished partial once: the dot is linear in the input. The squares are of
// the raw input. The last threadgroup's threads, in route_last, each add the
// splits of their rows, the group selects by rounds of its largest packed
// key, thread 0 writes the weights, and the ticket resets.

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
#define ROUTER_HOLD ((ROUTER_ROWS + ROUTER_THREADS - 1u) / ROUTER_THREADS)
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

// The last threadgroup's whole epilogue, on every thread of the group, after
// ticket_acquire: the split add in parallel, one row per strided thread, the
// selection by rounds of the threadgroup's largest packed key, and the
// weights and the reset on thread 0, serial, so the weights' bits stay put.
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
                       threadgroup ulong *stage,
                       threadgroup uint *flags)
{
    ticket_acquire();
    // Each row's thread adds its row's split 0 then split 1, the order thread
    // 0 once added them in, so every logit is the bits it was. The held keys
    // are ticket.metal's argmax_key of the same value, and the empty slots of
    // a short share hold key 0, which no packed key is, so they never win.
    float chosen_v[ROUTER_TOPK_MAX];
    ulong held[ROUTER_HOLD];
    uint held_n = 0u;
    uint mine_bad = 0u;
    for (uint row = tid; row < ROUTER_ROWS; row += ROUTER_THREADS) {
        float p0 = ticket_load(partials, row * ROUTER_SPLIT);
        float p1 = ticket_load(partials, row * ROUTER_SPLIT + 1u);
        float l = p0 + p1;
        logits[row] = l;
        tg_logits[row] = l;
        mine_bad |= isfinite(l) ? 0u : 1u;
        // The shared expert's row is not a selection candidate; the old scan
        // read only the routed experts.
        if (row < ROUTER_EXPERTS) {
            held[held_n++] = argmax_key(l, row);
        }
    }
    for (uint j = held_n; j < ROUTER_HOLD; ++j) {
        held[j] = 0u;
    }
    // The logits and the finiteness word cross the group here.
    mine_bad = simd_max(mine_bad);
    if (lane == 0u) {
        flags[sg_in_tg] = mine_bad;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint bad = flags[0];
    for (uint g = 1u; g < ROUTER_SG; ++g) {
        bad = max(bad, flags[g]);
    }
    if (bad != 0u) {
        if (tid == 0u) {
            raise(faults, args.layer, BLOOMERY_FAULT_SITE_ROUTER);
            for (uint s = 0u; s < args.top_k; ++s) {
                ids[s] = s;
                weights[s] = NAN;
            }
            ids[args.top_k] = ROUTER_EXPERTS;
            weights[args.top_k] = NAN;
            ticket_reset(ticket);
        }
        return;
    }
    // One round per selected expert: the group's largest key names this
    // round's selection, and the round's winner is keyed out of the next.
    // The key's order is the old scan's: the larger value first, at an equal
    // value the smaller id.
    for (uint s = 0u; s < args.top_k; ++s) {
        ulong local = 0u;
        for (uint j = 0u; j < held_n; ++j) {
            local = max(local, held[j]);
        }
        ulong chosen_key = ticket_max_key(local, stage, lane, sg_in_tg, ROUTER_SG);
        if (tid == 0u) {
            uint best_i = ~uint(chosen_key);
            ids[s] = best_i;
            chosen_v[s] = tg_logits[best_i];
        }
        for (uint j = 0u; j < held_n; ++j) {
            if (held[j] == chosen_key) {
                held[j] = 0u;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // The softmax and the shared expert's sigmoid stay on one thread, in the
    // order the serial epilogue once used, so the weights' bits do not move.
    if (tid == 0u) {
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
        ticket_reset(ticket);
    }
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
    threadgroup float tg_logits[ROUTER_ROWS];
    threadgroup ulong stage[ROUTER_SG];
    threadgroup uint flags[ROUTER_SG];

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
    if (last_flag == 1u) {
        route_last(partials, ticket, logits, ids, weights, faults, args,
                   tid, lane, sg_in_tg, tg_logits, stage, flags);
    }
}
