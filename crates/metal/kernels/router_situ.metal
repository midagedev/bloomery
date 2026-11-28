// The biased sigmoid router of Inkling 2: sigmoid scores over
// BLOOMERY_INK2_EXPERTS experts, a selection bias that steers the choice
// only (the kept weights are the unbiased scores), the kept scores
// renormalized by their sum plus the fixed 1e-20 and scaled by the launch's
// route scale — the registered router's own structure (a row a SIMD group,
// eight a threadgroup, the last threadgroup selecting by rounds of the
// group's largest packed key) at this family's rule.
//
// The registered instances' router is the softmax one (no bias, the kept
// scores over their sum, a hidden width a multiple of 1024 so the norm's
// lanes cover the input exactly); this kernel is the family's own source, a
// variant count its own. Its hidden width is a multiple of 128 — a float4 a
// lane a pass over the 32 lanes — and the input is the feed-forward mix's
// output as it is: this router norms nothing on load (the registered
// router's launch feeds a residual, the mix's output is not one).
//
// One dispatch: each threadgroup's eight rows dot their whole width (a
// lane's float4s, the four products of each added left to right), the
// ticket publishes the partials, and the last threadgroup reads every row's
// score back, packs the selection keys (the score plus the row's bias) and
// selects: one round a kept expert, the larger key first and the smaller id
// at an equal value, the same order the argmax key packs. Lane 0 writes the
// ids and the weights — the kept scores' sum in slot order, the divisor the
// sum plus the fixed 1e-20 (a guard under half an ulp of every sum a
// sigmoid's eight can make, an all-zero selection weighing 0), each weight
// the score over the divisor times the route scale, the scale applied after
// the norm.
//
// No silent failure: a score or a selection value that is not finite raises
// the router site and the weights are NaN over the ids 0.. in order, so the
// finite rank order is not a defined pick.
//
// This file comes after fault.metal (raise), router.metal (router_load4)
// and ticket.metal (the ticket and the argmax key).

#pragma METAL fp contract(off)

#define SI_EXPERTS BLOOMERY_INK2_EXPERTS
#define SI_K BLOOMERY_INK2_HIDDEN
#define SI_LANES 32u
#define SI_SG 8u
#define SI_THREADS (SI_LANES * SI_SG)
#define SI_KEYS (SI_EXPERTS / SI_LANES)
#define SI_FLOAT4S (SI_K / 4u / SI_LANES)
#define SI_TOPK BLOOMERY_INK2_TOP_K

static_assert(SI_K % 128u == 0u, "a float4 a lane a pass over the 32 lanes");
static_assert(SI_LANES * SI_FLOAT4S * 4u == SI_K, "the lanes cover the width exactly");
static_assert(SI_EXPERTS % SI_LANES == 0u, "the selection's lanes take whole rows of keys");
static_assert(SI_TOPK <= SI_EXPERTS, "the kept experts are among all of them");

// The launch's argument block, bound by value.
struct SigmoidArgs {
    uint top_k;   // the experts a token keeps
    uint layer;
    float scale;  // the route scale, the description's router scale
};

static_assert(sizeof(SigmoidArgs) == 12, "SigmoidArgs is 12 bytes: the host packs three words");

// The selection, from the scores and keys a threadgroup published: the
// first SIMD group alone, after the barrier that published them. The
// weights stay on lane 0, serial, so their bits stay put.
static void situ_pick(device uint *ids,
                      device float *weights,
                      device atomic_uint *ticket,
                      device atomic_uint *faults,
                      constant SigmoidArgs &args,
                      uint lane,
                      threadgroup float *tg_score,
                      threadgroup ulong *tg_keys,
                      threadgroup uint *tg_bad)
{
    uint bad = tg_bad[0];
    for (uint g = 1u; g < SI_SG; ++g) {
        bad = max(bad, tg_bad[g]);
    }
    if (bad != 0u) {
        if (lane == 0u) {
            raise(faults, args.layer, BLOOMERY_FAULT_SITE_ROUTER);
            for (uint s = 0u; s < args.top_k; ++s) {
                ids[s] = s;
                weights[s] = NAN;
            }
            ticket_reset(ticket);
        }
        return;
    }
    ulong held[SI_KEYS];
    for (uint j = 0u; j < SI_KEYS; ++j) {
        held[j] = tg_keys[lane + SI_LANES * j];
    }
    // One round a kept expert: the group's largest key names the round's
    // selection, the round's winner keyed out of the next.
    float chosen[SI_TOPK];
    for (uint s = 0u; s < args.top_k; ++s) {
        ulong local = 0u;
        for (uint j = 0u; j < SI_KEYS; ++j) {
            local = max(local, held[j]);
        }
        uint hi = simd_max(uint(local >> 32));
        uint lo = simd_max(uint(local >> 32) == hi ? uint(local) : 0u);
        const ulong chosen_key = ((ulong)hi << 32) | (ulong)lo;
        for (uint j = 0u; j < SI_KEYS; ++j) {
            held[j] = held[j] == chosen_key ? 0u : held[j];
        }
        if (lane == 0u) {
            const uint winner = ~uint(chosen_key);
            ids[s] = winner;
            chosen[s] = tg_score[winner];
        }
    }
    if (lane == 0u) {
        // The kept scores' sum in slot order, the divisor with the fixed
        // guard, each weight the score over it times the scale — the scale
        // after the norm, this order the gate.
        float sum = 0.f;
        for (uint s = 0u; s < args.top_k; ++s) {
            sum += chosen[s];
        }
        const float divisor = sum + 1e-20f;
        for (uint s = 0u; s < args.top_k; ++s) {
            weights[s] = chosen[s] / divisor * args.scale;
        }
        ticket_reset(ticket);
    }
}

// Everything after the rows' dots: the ticket publishes each SIMD group's
// score, the last threadgroup reads every row's score back, packs the
// selection keys (the score plus the row's bias), flags the rows that are
// not finite and selects. Both entries end here.
static void situ_finish(device coherent(device) float *partials,
                        device atomic_uint *ticket,
                        device const float *bias,
                        device float *logits,
                        device uint *ids,
                        device float *weights,
                        constant SigmoidArgs &args,
                        device atomic_uint *faults,
                        uint groups,
                        uint tid,
                        uint lane,
                        uint sg_in_tg,
                        threadgroup uint &last_flag,
                        threadgroup float *tg_score,
                        threadgroup ulong *tg_keys,
                        threadgroup uint *tg_bad)
{
    ticket_is_last(ticket, groups, tid, last_flag);
    if (last_flag != 1u) {
        return;
    }
    ticket_acquire();
    // Every thread reads its rows' scores back — the values one lane wrote
    // each, so every score is the bits it was — packs the selection keys
    // (the score plus the row's bias) and flags the rows that are not
    // finite.
    uint mine_bad = 0u;
    for (uint row = tid; row < SI_EXPERTS; row += SI_THREADS) {
        const float score = ticket_load(partials, row);
        logits[row] = score;
        tg_score[row] = score;
        const float sel = score + bias[row];
        mine_bad |= isfinite(score) && isfinite(sel) ? 0u : 1u;
        tg_keys[row] = argmax_key(sel, row);
    }
    mine_bad = simd_max(mine_bad);
    if (lane == 0u) {
        tg_bad[sg_in_tg] = mine_bad;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_in_tg != 0u) {
        return;
    }
    situ_pick(ids, weights, ticket, faults, args, lane, tg_score, tg_keys, tg_bad);
}

kernel void router_sigmoid(device const float *rows [[buffer(0)]],
                           device const float *x [[buffer(1)]],
                           device const float *bias [[buffer(2)]],
                           device coherent(device) float *partials [[buffer(3)]],
                           device atomic_uint *ticket [[buffer(4)]],
                           device float *logits [[buffer(5)]],
                           device uint *ids [[buffer(6)]],
                           device float *weights [[buffer(7)]],
                           constant SigmoidArgs &args [[buffer(8)]],
                           device atomic_uint *faults [[buffer(9)]],
                           uint tg [[threadgroup_position_in_grid]],
                           uint tid [[thread_position_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint sg_in_tg [[simdgroup_index_in_threadgroup]],
                           uint groups [[threadgroups_per_grid]])
{
    threadgroup uint last_flag = 0u;
    threadgroup float tg_score[SI_EXPERTS];
    threadgroup ulong tg_keys[SI_EXPERTS];
    threadgroup uint tg_bad[SI_SG];

    // The row this SIMD group owns, its dot over the whole width: a lane's
    // float4s of the row and the input, the four products of each added
    // left to right, the group's lanes summed.
    const uint sg = tg * SI_SG + sg_in_tg;
    if (sg < SI_EXPERTS) {
        device const float *row = rows + (ulong)sg * (ulong)SI_K;
        float acc = 0.f;
        for (uint i = 0u; i < SI_FLOAT4S; ++i) {
            const float4 w = router_load4(row, lane + SI_LANES * i);
            const float4 xv = router_load4(x, lane + SI_LANES * i);
            acc += w.x * xv.x;
            acc += w.y * xv.y;
            acc += w.z * xv.z;
            acc += w.w * xv.w;
        }
        // The score is the sigmoid of the logit; the selection value adds
        // the row's bias on top of it, steering the choice only.
        const float score = 1.0f / (1.0f + exp(-simd_sum(acc)));
        if (lane == 0u) {
            ticket_write(partials, sg, score);
        }
    }
    situ_finish(partials, ticket, bias, logits, ids, weights, args, faults, groups, tid, lane,
                sg_in_tg, last_flag, tg_score, tg_keys, tg_bad);
}

// The router with the feed-forward's norm in front of it: every threadgroup
// norms the input itself (norm_rows' own sums, the same threads, loops and
// SIMD-group tree, so every copy is bit-equal) and keeps the normed row in
// threadgroup memory for its rows' dots; threadgroup 0 also stores it, the
// experts' input. The norm raises the norm site as norm_rows does (a row it
// cannot take is NaN, and the scores of NaN then raise the router's own site).
// The rows' weights are loaded before the norm's sums so their latency runs
// under it.
struct SigmoidNormArgs {
    SigmoidArgs router;
    float eps;  // the norm's epsilon
};

static_assert(sizeof(SigmoidNormArgs) == 16, "SigmoidNormArgs is 16 bytes: the host packs four words");
static_assert(NORM_ROWS_THREADS == SI_THREADS, "the norm's threads are the router's");

kernel void router_sigmoid_norm(device const float *rows [[buffer(0)]],
                                device const float *x [[buffer(1)]],
                                device const float *gain [[buffer(2)]],
                                device float *normed [[buffer(3)]],
                                device const float *bias [[buffer(4)]],
                                device coherent(device) float *partials [[buffer(5)]],
                                device atomic_uint *ticket [[buffer(6)]],
                                device float *logits [[buffer(7)]],
                                device uint *ids [[buffer(8)]],
                                device float *weights [[buffer(9)]],
                                constant SigmoidNormArgs &args [[buffer(10)]],
                                device atomic_uint *faults [[buffer(11)]],
                                uint tg [[threadgroup_position_in_grid]],
                                uint tid [[thread_position_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]],
                                uint sg_in_tg [[simdgroup_index_in_threadgroup]],
                                uint groups [[threadgroups_per_grid]])
{
    threadgroup uint last_flag = 0u;
    threadgroup float tg_score[SI_EXPERTS];
    threadgroup ulong tg_keys[SI_EXPERTS];
    threadgroup uint tg_bad[SI_SG];
    threadgroup float tg_sums[SI_SG];
    threadgroup float4 tg_x[SI_K / 4u];

    // The row's weights, loaded first.
    const uint sg = tg * SI_SG + sg_in_tg;
    float4 wv[SI_FLOAT4S];
    if (sg < SI_EXPERTS) {
        device const float *row = rows + (ulong)sg * (ulong)SI_K;
        for (uint i = 0u; i < SI_FLOAT4S; ++i) {
            wv[i] = router_load4(row, lane + SI_LANES * i);
        }
    }

    // norm_rows' sums and value, line for line.
    const uint f4s = SI_K / 4u;
    float sum_sq = 0.f;
    for (uint f4 = tid; f4 < f4s; f4 += SI_THREADS) {
        float4 xv = norm_rows_load4(x, f4);
        sum_sq += xv.x * xv.x + xv.y * xv.y + xv.z * xv.z + xv.w * xv.w;
    }
    sum_sq = simd_sum(sum_sq);
    if (lane == 0u) {
        tg_sums[sg_in_tg] = sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.f;
    for (uint g = 0u; g < SI_SG; ++g) {
        total += tg_sums[g];
    }
    const float scale = rms_scale(total, SI_K, args.eps);
    const bool usable = rms_usable(total, scale);
    if (!usable && tg == 0u && tid == 0u) {
        norm_rows_raise(faults, args.router.layer);
    }
    bool bad = false;
    for (uint f4 = tid; f4 < f4s; f4 += SI_THREADS) {
        const float4 value = usable
            ? norm_rows_value(norm_rows_load4(x, f4), norm_rows_load4(gain, f4), scale)
            : float4(NAN);
        bad = bad || !all(isfinite(value));
        tg_x[f4] = value;
        if (tg == 0u) {
            *((device float4 *)normed + f4) = value;
        }
    }
    if (usable && bad && tg == 0u) {
        norm_rows_raise(faults, args.router.layer);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg < SI_EXPERTS) {
        float acc = 0.f;
        for (uint i = 0u; i < SI_FLOAT4S; ++i) {
            const float4 w = wv[i];
            const float4 xv = tg_x[lane + SI_LANES * i];
            acc += w.x * xv.x;
            acc += w.y * xv.y;
            acc += w.z * xv.z;
            acc += w.w * xv.w;
        }
        const float score = 1.0f / (1.0f + exp(-simd_sum(acc)));
        if (lane == 0u) {
            ticket_write(partials, sg, score);
        }
    }
    situ_finish(partials, ticket, bias, logits, ids, weights, args.router, faults, groups, tid,
                lane, sg_in_tg, last_flag, tg_score, tg_keys, tg_bad);
}
