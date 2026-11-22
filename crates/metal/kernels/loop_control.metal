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
// One launch covers one module of a core layer (the attention's, the k's,
// the v's or the feed-forward's: its own down, gain and forget rows over
// its own cur and msg rows), one threadgroup of 256 threads a row. A visit
// enqueues it once a module — the three attention modules' inputs are ready
// together after the message mixes, the feed-forward's only after the block
// join, so the four never share a launch.
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
// The staged [cur; msg] row is 2 · BLOOMERY_INK2_HIDDEN f32 of threadgroup
// memory, the Mini's width; the family's wider file does not fit a
// threadgroup at its staging and is refused at the host.
//
// Numeric order: the two sums of squares and the down and gain/forget dots
// accumulate by fused multiply-adds ascending their fast index, the SIMD
// group's lanes summed by simd_sum and its threadgroup's eight shares added
// in group order (the reduction order this backend's kernels share, a named
// band term against the reference's warp butterflies); the staged normed
// value v · rstd rounds on its own; the combine is one rounded multiply each
// and one rounded add. The tanh is the device's own.
//
// No silent failure: a half whose sum of squares is not usable (rms_usable,
// norm.metal) or an output that stops being finite raises the product site
// and the value written there is NaN, never a plausible merge.
//
// This file comes after fault.metal (raise) and norm.metal (rms_scale,
// rms_usable).

#pragma METAL fp contract(off)

#define LC_THREADS 256u
#define LC_SG (LC_THREADS / 32u)

static_assert(LC_THREADS == 256u, "the controller's threadgroup is the norm's own width");
static_assert(2u * BLOOMERY_INK2_HIDDEN * 4u <= 32u * 1024u,
              "the staged [cur; msg] row fits a threadgroup's memory");

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

// One row's controller, one threadgroup arrived converged.
static void loop_control_row(device const float *cur,
                             device const float *msg,
                             device const float *down,
                             device const float *gain,
                             device const float *forget,
                             device float *y,
                             constant LoopControlArgs &args,
                             device atomic_uint *faults,
                             uint t,
                             uint tid,
                             threadgroup float *staged,
                             threadgroup float *h,
                             threadgroup float *partials,
                             threadgroup float *rstd)
{
    const uint n = args.n;
    const uint rank = args.rank;
    const uint row = t * n;
    const uint sg = tid / 32u;
    const uint lane = tid % 32u;

    // Stage the raw row and its two sums of squares, a thread's share of
    // each half.
    float sq_c = 0.f;
    float sq_m = 0.f;
    for (uint j = tid; j < n; j += LC_THREADS) {
        const float c = cur[row + j];
        const float g = msg[row + j];
        sq_c = fma(c, c, sq_c);
        sq_m = fma(g, g, sq_m);
        staged[j] = c;
        staged[n + j] = g;
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
    if (isnan(rstd[0])) {
        raise(faults, uint(args.layer), BLOOMERY_FAULT_SITE_NORM_QUANT);
        for (uint j = tid; j < n; j += LC_THREADS) {
            y[row + j] = NAN;
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

    // The down-dots: SIMD group sg takes outputs sg, sg + LC_SG, ..., its
    // lanes the row's values lane, lane + 32, ..., ascending.
    for (uint r = sg; r < rank; r += LC_SG) {
        float acc = 0.f;
        for (uint k = lane; k < 2u * n; k += 32u) {
            acc = fma(down[r * 2u * n + k], staged[k], acc);
        }
        acc = simd_sum(acc);
        if (lane == 0u) {
            h[r] = acc;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // The gain and forget rows, the combine: each thread its output value.
    bool ok = true;
    for (uint j = tid; j < n; j += LC_THREADS) {
        float g = 0.f;
        float f = 0.f;
        for (uint d = 0u; d < rank; ++d) {
            g = fma(gain[j * rank + d], h[d], g);
            f = fma(forget[j * rank + d], h[d], f);
        }
        const float out = (1.0f - tanh(f)) * cur[row + j] + tanh(g) * msg[row + j];
        ok = ok && isfinite(out);
        y[row + j] = isfinite(out) ? out : NAN;
    }
    if (!ok) {
        raise(faults, args.layer, BLOOMERY_FAULT_SITE_F32_PRODUCT);
    }
}

kernel void loop_control(device const float *cur [[buffer(0)]],
                         device const float *msg [[buffer(1)]],
                         device const float *down [[buffer(2)]],
                         device const float *gain [[buffer(3)]],
                         device const float *forget [[buffer(4)]],
                         device float *y [[buffer(5)]],
                         constant LoopControlArgs &args [[buffer(6)]],
                         device atomic_uint *faults [[buffer(7)]],
                         uint t [[threadgroup_position_in_grid]],
                         uint tid [[thread_position_in_threadgroup]])
{
    if (t >= args.m) {
        return;
    }
    threadgroup float staged[2 * BLOOMERY_INK2_HIDDEN];
    threadgroup float h[BLOOMERY_INK2_RANK];
    threadgroup float partials[2u * LC_SG];
    threadgroup float rstd[2];
    loop_control_row(cur, msg, down, gain, forget, y, args, faults, t, tid, staged, h, partials,
                     rstd);
}
