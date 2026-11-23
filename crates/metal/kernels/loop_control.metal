// The looped core's message controller of Inkling 2: the merge of a message
// mixed from the previous pass's banked blocks into a sublayer's input,
//
//   out = (1 − tanh(F·h)) · cur + tanh(G·h) · msg,   h = D · [rms(cur); rms(msg)]
//
// the two halves gainless-RMS-normed on their own at eps, the whole row
// down-projected to the gate rank, and the gain G and forget F rows of the
// rank writing the hidden row back. The weights are f32 in the file and the
// reference computes f32 throughout; so does the card.
//
// A module of a core layer (the attention's, the k's, the v's or the
// feed-forward's: its own down, gain and forget rows over its own cur and
// msg rows) is two launches. The down launch (loop_control_down) writes
// h, BLOOMERY_INK2_RANK / 8 threadgroups of 256 threads a row, a SIMD group
// an output of the rank; the merge (loop_control) reads it back, a thread
// an output value, ⌈n / 256⌉ threadgroups a row. Each threadgroup of either
// launch sums the row's two halves' squares itself, in the same order, so
// every one holds the same scales. A visit enqueues the three attention
// modules' down launches together and their merges together after them —
// their inputs are ready at once after the message mixes — and the
// feed-forward's only after the block join.
//
// Layouts, in the file's own memory order, no repack:
//   - cur and msg [m][n], y [m][n], token-major;
//   - down [rank][2n]: output r's 2n weights contiguous, the input k of
//     [cur; msg] fastest — a SIMD group's lanes read consecutive k, one dot
//     a group;
//   - gain and forget [n][rank]: output j's rank weights contiguous at
//     j · rank, the rank fastest — a thread's dot reads its own output's
//     row.
//
// The down launch stages the normed [cur; msg] row, 2 · BLOOMERY_INK2_HIDDEN
// f32 of threadgroup memory, the Mini's width; the family's wider file does
// not fit a threadgroup at its staging and is refused at the host.
//
// Rows: m rows are the grid's height, threadgroup (b, t) row t's; every row
// reads the same weights and its own cur, msg, h and y row, so a row's bits
// are its m = 1 launch's.
//
// Numeric order: the two sums of squares and the down and gain/forget dots
// accumulate by fused multiply-adds ascending their fast index, the SIMD
// group's lanes summed by simd_sum and its threadgroup's eight shares added
// in group order (the reduction order this backend's kernels share, a named
// band term against the reference's warp butterflies); the staged normed
// value v · rstd rounds on its own; the combine is one rounded multiply each
// and one rounded add. The tanh is the device's own.
//
// Latency: a thread loads its LC_PER_THREAD values of each half, a lane
// LC_DOWN_PREFETCH of its down weights and a thread LC_MERGE_PREFETCH of its
// gain and forget weights before it adds the first of them, so a row's loads
// are in flight together instead of one after another. The loads moved, not
// the sums: every accumulation above keeps its order.
//
// No silent failure: a half whose sum of squares is not usable (rms_usable,
// norm.metal) raises the norm site and the row's h and outputs are NaN; an
// output that stops being finite raises the product site and the value
// written there is NaN, never a plausible merge.
//
// This file comes after fault.metal (raise) and norm.metal (rms_scale,
// rms_usable).

#pragma METAL fp contract(off)

#define LC_THREADS 256u
#define LC_SG (LC_THREADS / 32u)
// A thread's values of each half loaded at once: the Mini's nine, one pass.
#define LC_PER_THREAD ((BLOOMERY_INK2_HIDDEN + LC_THREADS - 1u) / LC_THREADS)
// A lane's down weights loaded at once: the Mini's 144 a lane in three.
#define LC_DOWN_PREFETCH 48u
// A thread's gain and forget weights loaded at once, whole packed float4s:
// the rank's 64 in two.
#define LC_MERGE_PREFETCH 32u

static_assert(LC_THREADS == 256u, "the controller's threadgroup is the norm's own width");
static_assert(2u * BLOOMERY_INK2_HIDDEN * 4u <= 32u * 1024u,
              "the staged [cur; msg] row fits a threadgroup's memory");
static_assert(BLOOMERY_INK2_RANK % LC_SG == 0u, "a down threadgroup's SIMD groups cover whole outputs");
static_assert(BLOOMERY_INK2_RANK % LC_MERGE_PREFETCH == 0u, "the merge's passes cover the rank");
static_assert(LC_MERGE_PREFETCH % 4u == 0u, "a merge pass is whole float4s");

// The launch's argument block, bound by value. The rank is the family's
// own; the width the staged staging's.
struct LoopControlArgs {
    uint n;     // the hidden width, BLOOMERY_INK2_HIDDEN
    uint rank;  // the gate rank, BLOOMERY_INK2_RANK
    uint m;     // rows
    uint layer;
    float eps;
};

static_assert(sizeof(LoopControlArgs) == 20, "LoopControlArgs is 20 bytes: the host packs five words");

// The two halves' RMS scales of row `row`, every thread of a threadgroup
// arrived converged; rstd[0] is NaN when either half is not usable. A thread
// sums its values tid, tid + 256, ... of each half; with `stage` it also
// stores the raw row, [cur; msg], at `staged`.
static void loop_control_scales(device const float *cur,
                                device const float *msg,
                                constant LoopControlArgs &args,
                                uint row,
                                uint tid,
                                bool stage,
                                threadgroup float *staged,
                                threadgroup float *partials,
                                threadgroup float *rstd)
{
    const uint n = args.n;
    const uint sg = tid / 32u;
    const uint lane = tid % 32u;
    float sq_c = 0.f;
    float sq_m = 0.f;
    for (uint j0 = tid; j0 < n; j0 += LC_THREADS * LC_PER_THREAD) {
        float c[LC_PER_THREAD];
        float g[LC_PER_THREAD];
        for (uint i = 0u; i < LC_PER_THREAD; ++i) {
            const uint j = j0 + LC_THREADS * i;
            c[i] = 0.f;
            g[i] = 0.f;
            if (j < n) {
                c[i] = cur[row + j];
                g[i] = msg[row + j];
            }
        }
        for (uint i = 0u; i < LC_PER_THREAD; ++i) {
            const uint j = j0 + LC_THREADS * i;
            if (j < n) {
                sq_c = fma(c[i], c[i], sq_c);
                sq_m = fma(g[i], g[i], sq_m);
                if (stage) {
                    staged[j] = c[i];
                    staged[n + j] = g[i];
                }
            }
        }
    }
    sq_c = simd_sum(sq_c);
    sq_m = simd_sum(sq_m);
    if (lane == 0u) {
        partials[2u * sg] = sq_c;
        partials[2u * sg + 1u] = sq_m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        // The eight shares in group order, the fixed tree this backend's
        // kernels reduce their threadgroups by.
        float sums[2] = {0.f, 0.f};
        for (uint g = 0u; g < LC_SG; ++g) {
            sums[0] += partials[2u * g];
            sums[1] += partials[2u * g + 1u];
        }
        const float scale_c = rms_scale(sums[0], n, args.eps);
        const float scale_m = rms_scale(sums[1], n, args.eps);
        if (!rms_usable(sums[0], scale_c) || !rms_usable(sums[1], scale_m)) {
            rstd[0] = NAN;
        } else {
            rstd[0] = scale_c;
            rstd[1] = scale_m;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// The down launch: threadgroup (b, t) writes h[t][8b .. 8b + 8), SIMD group
// sg output 8b + sg, its lanes the normed row's values lane, lane + 32, ...,
// ascending, LC_DOWN_PREFETCH weights of a lane loaded before they are
// added.
kernel void loop_control_down(device const float *cur [[buffer(0)]],
                              device const float *msg [[buffer(1)]],
                              device const float *down [[buffer(2)]],
                              device const float *gain [[buffer(3)]],
                              device const float *forget [[buffer(4)]],
                              device float *y [[buffer(5)]],
                              constant LoopControlArgs &args [[buffer(6)]],
                              device atomic_uint *faults [[buffer(7)]],
                              device float *h [[buffer(8)]],
                              uint2 tg [[threadgroup_position_in_grid]],
                              uint tid [[thread_index_in_threadgroup]])
{
    const uint t = tg.y;
    if (t >= args.m) {
        return;
    }
    threadgroup float staged[2 * BLOOMERY_INK2_HIDDEN];
    threadgroup float partials[2u * LC_SG];
    threadgroup float rstd[2];
    const uint n = args.n;
    const uint sg = tid / 32u;
    const uint lane = tid % 32u;
    const uint r = tg.x * LC_SG + sg;
    loop_control_scales(cur, msg, args, t * n, tid, true, staged, partials, rstd);
    if (isnan(rstd[0])) {
        if (tid == 0u) {
            raise(faults, uint(args.layer), BLOOMERY_FAULT_SITE_NORM_QUANT);
        }
        if (lane == 0u && r < args.rank) {
            h[t * args.rank + r] = NAN;
        }
        return;
    }

    // Scale the staged halves in place: the normed values the down-dots
    // read.
    for (uint j = tid; j < n; j += LC_THREADS) {
        staged[j] = staged[j] * rstd[0];
        staged[n + j] = staged[n + j] * rstd[1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (r >= args.rank) {
        return;
    }
    const uint kn = 2u * n;
    device const float *w = down + r * 2u * n;
    float acc = 0.f;
    for (uint k0 = lane; k0 < kn; k0 += 32u * LC_DOWN_PREFETCH) {
        float wv[LC_DOWN_PREFETCH];
        for (uint u = 0u; u < LC_DOWN_PREFETCH; ++u) {
            const uint k = k0 + 32u * u;
            wv[u] = k < kn ? w[k] : 0.f;
        }
        for (uint u = 0u; u < LC_DOWN_PREFETCH; ++u) {
            const uint k = k0 + 32u * u;
            if (k < kn) {
                acc = fma(wv[u], staged[k], acc);
            }
        }
    }
    acc = simd_sum(acc);
    if (lane == 0u) {
        h[t * args.rank + r] = acc;
    }
}

// The merge: thread tid of threadgroup (b, t) writes output 256b + tid of
// row t from the gain and forget rows and the row's h, LC_MERGE_PREFETCH of
// each row's weights loaded before they are added.
kernel void loop_control(device const float *cur [[buffer(0)]],
                         device const float *msg [[buffer(1)]],
                         device const float *down [[buffer(2)]],
                         device const float *gain [[buffer(3)]],
                         device const float *forget [[buffer(4)]],
                         device float *y [[buffer(5)]],
                         constant LoopControlArgs &args [[buffer(6)]],
                         device atomic_uint *faults [[buffer(7)]],
                         device const float *h [[buffer(8)]],
                         uint2 tg [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]])
{
    const uint t = tg.y;
    if (t >= args.m) {
        return;
    }
    threadgroup float hs[BLOOMERY_INK2_RANK];
    threadgroup float partials[2u * LC_SG];
    threadgroup float rstd[2];
    const uint n = args.n;
    const uint rank = args.rank;
    const uint row = t * n;
    const uint j = tg.x * LC_THREADS + tid;
    loop_control_scales(cur, msg, args, row, tid, false, hs, partials, rstd);
    if (isnan(rstd[0])) {
        if (tid == 0u) {
            raise(faults, uint(args.layer), BLOOMERY_FAULT_SITE_NORM_QUANT);
        }
        if (j < n) {
            y[row + j] = NAN;
        }
        return;
    }
    for (uint d = tid; d < rank; d += LC_THREADS) {
        hs[d] = h[t * rank + d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (j >= n) {
        return;
    }
    const float c = cur[row + j];
    const float m = msg[row + j];
    // The host holds the rank at BLOOMERY_INK2_RANK, whole passes of float4s;
    // a packed float4 reads on a float's alignment.
    device const packed_float4 *gw = (device const packed_float4 *)(gain + j * rank);
    device const packed_float4 *fw = (device const packed_float4 *)(forget + j * rank);
    float g = 0.f;
    float f = 0.f;
    for (uint d0 = 0u; d0 < rank; d0 += LC_MERGE_PREFETCH) {
        float4 gv[LC_MERGE_PREFETCH / 4u];
        float4 fv[LC_MERGE_PREFETCH / 4u];
        for (uint u = 0u; u < LC_MERGE_PREFETCH / 4u; ++u) {
            gv[u] = float4(gw[d0 / 4u + u]);
            fv[u] = float4(fw[d0 / 4u + u]);
        }
        for (uint u = 0u; u < LC_MERGE_PREFETCH / 4u; ++u) {
            for (uint e = 0u; e < 4u; ++e) {
                const float hd = hs[d0 + 4u * u + e];
                g = fma(gv[u][e], hd, g);
                f = fma(fv[u][e], hd, f);
            }
        }
    }
    const float out = (1.0f - tanh(f)) * c + tanh(g) * m;
    if (!isfinite(out)) {
        raise(faults, args.layer, BLOOMERY_FAULT_SITE_F32_PRODUCT);
    }
    y[row + j] = isfinite(out) ? out : NAN;
}
