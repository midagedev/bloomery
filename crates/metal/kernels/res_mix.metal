// The block attention residual's normed mix of Inkling 2: the vector a
// sublayer reads is a softmax-weighted sum of the banked block inputs and
// the running stream, every state RMS-normed at a fixed 1e-6 and the output
// summing the normed states — one launch, one threadgroup of 256 threads a
// mix point, up to BLOOMERY_INK2_MIX_VECS vectors and BLOOMERY_INK2_MIX_NQ
// query rows in the one staging.
//
// For a row t, the V = n_banks + 1 vectors are the bank rows bank_i[t] and
// the stream row cur[t]. Each vector v is scored, a query row q, by
// s = (rstd(v) · dot(v, q)) · scale — rstd the shared RMS scale
// 1/sqrt(mean(v²) + eps), scale the launch word 1/sqrt(hidden) the caller
// derives — and each query row's scores go through one softmax over the V
// vectors; the output row q is Σ_i p_i · (v_i · rstd_i), the normed value
// rounding on its own before the fused multiply-add, as the reference's
// materialized normed states do.
//
// Layout: bank i's row t is banks[i * stride + t * n ..][..n], the stream's
// and the outputs' row t is cur[t * n ..] and y[t * nq * n + q * n ..];
// stride >= m * n, so a threadgroup never reads another row.
//
// Numeric order: a SIMD group takes a vector at a time (its groups
// striding the vectors), each lane the values lane, lane + 32, ..., its
// query dots and its squares accumulated in f32 in that order, the group
// summing its lanes (simd_sum, the norm's own order in norm.metal). The
// reference's threads walk the whole row and reduce eight warps in a fixed
// tree; this order is the named band term between the two. The softmax is
// exp(s - max) in f32, summed in vector order, each divided by the sum.
//
// No silent failure: a vector whose sum of squares is not usable
// (rms_usable, norm.metal) or a score that is not finite raises the norm
// site and the query row's output is NaN, never a plausible mix.
//
// This file comes after fault.metal (raise) and norm.metal (rms_scale,
// rms_usable).

#pragma METAL fp contract(off)

#define MIX_THREADS 256u
#define MIX_SG (MIX_THREADS / 32u)

static_assert(MIX_THREADS == 256u, "the mix's threadgroup is the norm's own width");

// The launch's argument block, bound by value.
struct ResMixArgs {
    uint n;        // values a vector
    uint nq;       // query rows the launch scores, 1..BLOOMERY_INK2_MIX_NQ
    uint m;        // mix points (rows)
    uint n_banks;  // the banks; the stream is their last vector
    uint stride;   // floats between banks' rows
    uint layer;
    float eps;
    float scale;   // the softmax's score scale, 1/sqrt(n) the caller derives
};

static_assert(sizeof(ResMixArgs) == 32, "ResMixArgs is 32 bytes: the host packs eight words");

// One threadgroup's staging: the probabilities, a row a query, and the
// vectors' shared RMS scales.
struct ResMixShared {
    float prob[BLOOMERY_INK2_MIX_NQ][BLOOMERY_INK2_MIX_VECS];
    float rstd[BLOOMERY_INK2_MIX_VECS];
};

// The mix of one row, one threadgroup of MIX_THREADS threads arrived
// converged. A SIMD group takes a vector at a time; thread 0 takes the
// softmax; the threadgroup sums the weighted normed states.
static void res_mix_row(device const float *banks,
                        device const float *cur,
                        device const float *queries,
                        device float *y,
                        constant ResMixArgs &args,
                        device atomic_uint *faults,
                        uint t,
                        uint tid,
                        threadgroup ResMixShared &shared)
{
    const uint n = args.n;
    const uint nq = args.nq;
    const uint vecs = args.n_banks + 1u;
    const uint row = t * n;
    const uint sg = tid / 32u;
    const uint lane = tid % 32u;
    for (uint v = sg; v < vecs; v += MIX_SG) {
        device const float *src =
            v < args.n_banks ? banks + v * args.stride + row : cur + row;
        float dot[BLOOMERY_INK2_MIX_NQ];
        for (uint q = 0u; q < nq; ++q) {
            dot[q] = 0.f;
        }
        float sq = 0.f;
        for (uint j = lane; j < n; j += 32u) {
            const float value = src[j];
            sq = fma(value, value, sq);
            for (uint q = 0u; q < nq; ++q) {
                dot[q] = fma(value, queries[q * n + j], dot[q]);
            }
        }
        for (uint q = 0u; q < nq; ++q) {
            dot[q] = simd_sum(dot[q]);
        }
        sq = simd_sum(sq);
        if (lane == 0u) {
            for (uint q = 0u; q < nq; ++q) {
                shared.prob[q][v] = dot[q];
            }
            shared.rstd[v] = sq;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        bool usable = true;
        for (uint v = 0u; v < vecs; ++v) {
            const float scale = rms_scale(shared.rstd[v], n, args.eps);
            usable = usable && rms_usable(shared.rstd[v], scale);
            shared.rstd[v] = scale;
        }
        for (uint q = 0u; q < nq; ++q) {
            float mx = -INFINITY;
            for (uint v = 0u; v < vecs; ++v) {
                // The dot rounds, then the rstd, then the scale (the
                // reference's mul_mat, then its soft_max's in-place scale);
                // this order is the gate.
                const float s = shared.rstd[v] * shared.prob[q][v] * args.scale;
                shared.prob[q][v] = s;
                mx = max(mx, s);
            }
            float total = 0.f;
            for (uint v = 0u; v < vecs; ++v) {
                const float e = exp(shared.prob[q][v] - mx);
                shared.prob[q][v] = e;
                total += e;
            }
            const float p = 1.f / total;
            for (uint v = 0u; v < vecs; ++v) {
                shared.prob[q][v] = usable ? shared.prob[q][v] * p : NAN;
            }
        }
        if (!usable) {
            raise(faults, uint(args.layer), BLOOMERY_FAULT_SITE_NORM_QUANT);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // The weighted sum of the normed states, a query row an output row.
    for (uint j = tid; j < n; j += MIX_THREADS) {
        for (uint q = 0u; q < nq; ++q) {
            float acc = 0.f;
            for (uint v = 0u; v < vecs; ++v) {
                device const float *src =
                    v < args.n_banks ? banks + v * args.stride + row : cur + row;
                const float normed = src[j] * shared.rstd[v];
                acc = fma(shared.prob[q][v], normed, acc);
            }
            y[t * nq * n + q * n + j] = acc;
        }
    }
}

kernel void res_mix_normed(device const float *banks [[buffer(0)]],
                           device const float *cur [[buffer(1)]],
                           device const float *queries [[buffer(2)]],
                           device float *y [[buffer(3)]],
                           constant ResMixArgs &args [[buffer(4)]],
                           device atomic_uint *faults [[buffer(5)]],
                           uint t [[threadgroup_position_in_grid]],
                           uint tid [[thread_position_in_threadgroup]])
{
    if (t >= args.m) {
        return;
    }
    threadgroup ResMixShared shared;
    res_mix_row(banks, cur, queries, y, args, faults, t, tid, shared);
}
