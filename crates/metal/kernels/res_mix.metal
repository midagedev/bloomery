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
// Latency: a lane loads MIX_PREFETCH of its values (and their query values)
// before it adds the first, and a thread MIX_COLS of its outputs' vector
// values before it sums the first, so a row's loads are in flight together
// instead of one after another; thread v takes vector v's RMS scale and
// thread q query row q's softmax. The loads moved, not the sums: every
// accumulation above keeps its order.
// The loops over those arrays unroll in full (FOR_UNROLL, quant.metal): an
// array indexed by a rolled loop's variable does not stay in registers.
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
// A lane's values loaded at once in the scores: the Mini's 72 a lane in six.
#define MIX_PREFETCH 12u
// A thread's outputs whose vector values are loaded at once in the sum: the
// Mini's nine a thread in three.
#define MIX_COLS 3u

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
// converged. A SIMD group takes a vector at a time; thread v takes vector
// v's scale and thread q query row q's softmax; the threadgroup sums the
// weighted normed states.
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
        FOR_UNROLL(uint q = 0u; q < BLOOMERY_INK2_MIX_NQ; ++q) {
            dot[q] = 0.f;
        }
        float sq = 0.f;
        for (uint j0 = lane; j0 < n; j0 += 32u * MIX_PREFETCH) {
            float value[MIX_PREFETCH];
            float query[MIX_PREFETCH][BLOOMERY_INK2_MIX_NQ];
            FOR_UNROLL(uint u = 0u; u < MIX_PREFETCH; ++u) {
                const uint j = j0 + 32u * u;
                value[u] = 0.f;
                FOR_UNROLL(uint q = 0u; q < BLOOMERY_INK2_MIX_NQ; ++q) {
                    query[u][q] = 0.f;
                    if (j < n && q < nq) {
                        query[u][q] = queries[q * n + j];
                    }
                }
                if (j < n) {
                    value[u] = src[j];
                }
            }
            FOR_UNROLL(uint u = 0u; u < MIX_PREFETCH; ++u) {
                if (j0 + 32u * u < n) {
                    sq = fma(value[u], value[u], sq);
                    FOR_UNROLL(uint q = 0u; q < BLOOMERY_INK2_MIX_NQ; ++q) {
                        if (q < nq) {
                            dot[q] = fma(value[u], query[u][q], dot[q]);
                        }
                    }
                }
            }
        }
        FOR_UNROLL(uint q = 0u; q < BLOOMERY_INK2_MIX_NQ; ++q) {
            if (q < nq) {
                dot[q] = simd_sum(dot[q]);
            }
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
    // A vector's scale, NaN where its sum of squares is not usable: a
    // usable scale is finite, so a NaN names the vector to every thread.
    if (tid < vecs) {
        const uint v = tid;
        const float scale = rms_scale(shared.rstd[v], n, args.eps);
        shared.rstd[v] = rms_usable(shared.rstd[v], scale) ? scale : NAN;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool usable = true;
    for (uint v = 0u; v < vecs; ++v) {
        usable = usable && !isnan(shared.rstd[v]);
    }
    if (tid < nq) {
        const uint q = tid;
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
    if (tid == 0u && !usable) {
        raise(faults, uint(args.layer), BLOOMERY_FAULT_SITE_NORM_QUANT);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // The weighted sum of the normed states, a query row an output row.
    for (uint j0 = tid; j0 < n; j0 += MIX_THREADS * MIX_COLS) {
        float state[MIX_COLS][BLOOMERY_INK2_MIX_VECS];
        FOR_UNROLL(uint c = 0u; c < MIX_COLS; ++c) {
            const uint j = j0 + MIX_THREADS * c;
            FOR_UNROLL(uint v = 0u; v < BLOOMERY_INK2_MIX_VECS; ++v) {
                state[c][v] = 0.f;
                if (j < n && v < vecs) {
                    device const float *src =
                        v < args.n_banks ? banks + v * args.stride + row : cur + row;
                    state[c][v] = src[j];
                }
            }
        }
        FOR_UNROLL(uint c = 0u; c < MIX_COLS; ++c) {
            const uint j = j0 + MIX_THREADS * c;
            if (j >= n) {
                break;
            }
            for (uint q = 0u; q < nq; ++q) {
                float acc = 0.f;
                FOR_UNROLL(uint v = 0u; v < BLOOMERY_INK2_MIX_VECS; ++v) {
                    if (v < vecs) {
                        const float normed = state[c][v] * shared.rstd[v];
                        acc = fma(shared.prob[q][v], normed, acc);
                    }
                }
                y[t * nq * n + q * n + j] = acc;
            }
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
