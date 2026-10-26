// The Mamba-2 decode kernels: the step of one token (the causal conv with its
// bias and SiLU, the selective scan at state 128, the D skip and the z gate)
// and the grouped RMS norm over its output.
//
// The rule, per token at position p, head h (n_head heads of head_dim
// channels, n_group groups of per = n_head / n_group heads that share one B
// and one C, conv_dim = d_inner + 2 · n_group · 128 conv channels):
//
//   [x; B; C] = SiLU(conv4(xbc) + bias)     depthwise causal conv of the last 4 inputs
//   dt        = dt_raw + dt_bias            σ = dt                 if dt > 20
//                                              = log1p(exp(dt))    otherwise
//   dA        = exp(σ · A)                  A and D: one float a head
//   s[n]      = s[n] · dA + B[n] · (x · σ)  the scan, state n = 0..128 of a channel
//   y         = Σ_n s[n] · C[n]             lane l's four n = l + 32 j, then the
//                                           butterfly 16, 8, 4, 2, 1
//   y         = y + x · D, then y = SiLU(z) · y
//   out       = gain · y / sqrt(mean(y²) + eps)   per group of d_inner / n_group
//
// Layouts, all f32: xbc [conv_dim] (x, then B per group, then C per group), z
// and y [d_inner] (channel c + head · head_dim), dt, dt_bias, A, D [n_head],
// the conv weight [conv_dim][4] (tap fastest, tap 0 the oldest input), the
// conv bias [conv_dim], the norm gain [d_inner].
//
// The scan state is [lanes][d_inner][128], n fastest. The step reads the lane
// the word at lane_at names and writes the lane the word at lane_to names,
// as the delta step does: the two words may name one lane, which the step
// then updates in place; two lanes apart, the lane it read is left as it was.
// The conv ring is [RING_ROWS][conv_dim], the delta step's ring: the input of
// position p lives in slot p mod RING_ROWS, and a predecessor at distance d
// is 0 while p < d, otherwise the slot (p − d) mod RING_ROWS. The slot a token
// writes is none of the slots it reads.
//
// The step's grid is n_head · chunks threadgroups of 128 threads, chunks =
// ceil(head_dim / 16): a threadgroup takes 16 channels of one head, a SIMD
// group four of them. Threadgroup b is head b / chunks, chunk b % chunks, so a
// group's threadgroups are contiguous. A threadgroup first stages its conv
// outputs in threadgroup memory (its group's 128 B and 128 C channels, which
// every threadgroup of the group computes again, and its own x channels) and
// writes the ring slots it owns: its x channels, and a group's B and C from
// the group's first threadgroup. A lane holds state n = lane + 32 j of each of
// four channels and its B and C values.
//
// The norm needs a whole group's y, which no one threadgroup holds: the step
// writes each channel's y (after the gate) and each threadgroup's sum of
// squares, in channel order, to part[b]; mamba2_norm is a second dispatch with
// one threadgroup a group, which sums the group's partials in threadgroup
// order and writes gain · scale · y.
//
// Faults. A dt, bias, A or D that is not finite, or a dt plus bias that
// overflows, raises the Mamba-2 scan site and writes nothing. A lane word at
// or past the lanes raises the lane site and writes nothing. A channel whose y
// is not finite raises the scan site, writes the y as it is, and leaves its
// state unwritten. A norm group whose mean square is not usable, and a
// normalized value that is not finite, raise the scan site; an unusable group
// is NaN.
//
// This file comes after delta.metal, whose butterfly sum, conv ring and
// state-lane scheme the step follows, norm.metal's scale helpers and
// fault.metal.

#include <metal_stdlib>

using namespace metal;

#pragma METAL fp contract(off)

#define MAMBA2_STATE 128u
#define MAMBA2_TAPS 4u
#define MAMBA2_THREADS 128u
#define MAMBA2_CH_PER_SIMD_GROUP 4u
#define MAMBA2_CH_PER_BLOCK 16u
// The staged conv outputs: the group's B and C, then the block's x.
#define MAMBA2_STAGED (2u * MAMBA2_STATE + MAMBA2_CH_PER_BLOCK)

static_assert(MAMBA2_TAPS == 4u, "the conv reads three predecessors and the current input");
static_assert(MAMBA2_STATE == 4u * 32u, "a lane holds four states");
static_assert(MAMBA2_THREADS == 128u && MAMBA2_CH_PER_BLOCK == 4u * MAMBA2_CH_PER_SIMD_GROUP,
              "four SIMD groups of four channels each");
static_assert(BLOOMERY_DELTA_RING_ROWS >= MAMBA2_TAPS, "the ring holds the conv's predecessors");

// The step's argument block, bound by value. Nine words.
struct Mamba2Args {
    uint n_head;
    uint n_group;
    uint head_dim;
    uint chunks;
    uint lanes;
    uint lane_at; // the word of the lane the step reads
    uint lane_to; // the word of the lane the step writes
    uint layer;
    uint pos;
};

static_assert(sizeof(Mamba2Args) == 36, "Mamba2Args is 36 bytes: the host packs nine words");

// The norm's argument block, bound by value. Six words: the epsilon is a float.
struct Mamba2NormArgs {
    uint n_head;
    uint n_group;
    uint head_dim;
    uint chunks;
    uint layer;
    float eps;
};

static_assert(sizeof(Mamba2NormArgs) == 24, "Mamba2NormArgs is 24 bytes: the host packs six words");

inline float mamba2_silu(float x) {
    return x / (1.0f + exp(-x));
}

// log(1 + x) for x >= 0, kept accurate where 1 + x rounds to 1 (and the
// plain log(1 + x) is 0 or a coarse step): with u = 1 + x rounded, the
// quotient x / (u - 1) cancels u's rounding error.
inline float mamba2_log1p(float x) {
    float u = 1.0f + x;
    return u == 1.0f ? x : log(u) * (x / (u - 1.0f));
}

// Whether channel k of chunk `chunk` exists in a head of `hd` channels.
inline bool mamba2_live(uint chunk, uint k, uint hd) {
    return chunk * MAMBA2_CH_PER_BLOCK + k < hd;
}

// The conv input `back` positions before `pos` in channel c: 0 while the
// position is before the sequence's first.
inline float mamba2_tap(device const float *ring, uint pos, uint back, uint conv_dim, uint c) {
    if (pos < back) {
        return 0.0f;
    }
    return ring[((pos + BLOOMERY_DELTA_RING_ROWS - back) % BLOOMERY_DELTA_RING_ROWS) * conv_dim + c];
}

// SiLU(conv4 + bias) of channel c: the products and sums in tap order, each
// rounded.
inline float mamba2_conv(float x0, float x1, float x2, float x3, device const float *w,
                         device const float *bias, uint c) {
    float acc = x0 * w[4u * c];
    acc += x1 * w[4u * c + 1u];
    acc += x2 * w[4u * c + 2u];
    acc += x3 * w[4u * c + 3u];
    return mamba2_silu(acc + bias[c]);
}

kernel void mamba2_step(device const float *xbc [[buffer(0)]],
                        device const float *dt [[buffer(1)]],
                        device const float *dt_bias [[buffer(2)]],
                        device const float *z [[buffer(3)]],
                        device const float *a [[buffer(4)]],
                        device const float *d [[buffer(5)]],
                        device const float *conv_w [[buffer(6)]],
                        device const float *conv_b [[buffer(7)]],
                        device const uint *lane_buf [[buffer(8)]],
                        device float *state [[buffer(9)]],
                        device float *ring [[buffer(10)]],
                        device float *y [[buffer(11)]],
                        device float *part [[buffer(12)]],
                        constant Mamba2Args &args [[buffer(13)]],
                        device atomic_uint *faults [[buffer(14)]],
                        uint tg [[threadgroup_position_in_grid]],
                        uint tid [[thread_position_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float stage[MAMBA2_STAGED];
    threadgroup float ys[MAMBA2_CH_PER_BLOCK];

    const uint hd = args.head_dim;
    const uint chunks = args.chunks;
    const uint per = args.n_head / args.n_group;
    const uint inner = args.n_head * hd;
    const uint conv_dim = inner + 2u * args.n_group * MAMBA2_STATE;
    const uint h = tg / chunks;
    const uint chunk = tg % chunks;
    const uint g = h / per;
    const uint pos = args.pos;

    // The head's scalars, the same in every thread. A non-finite one is a
    // fault before anything is written: a -inf dt or A would otherwise decay
    // to a finite y.
    const float dt_h = dt[h];
    const float bias_h = dt_bias[h];
    const float a_h = a[h];
    const float d_h = d[h];
    const float dtv = dt_h + bias_h;
    if (!(isfinite(dt_h) && isfinite(bias_h) && isfinite(a_h) && isfinite(d_h) && isfinite(dtv))) {
        if (tid == 0u) {
            raise(faults, args.layer, BLOOMERY_FAULT_SITE_MAMBA2_SCAN);
        }
        return;
    }

    const uint read_word = lane_buf[args.lane_at];
    const uint write_word = lane_buf[args.lane_to];
    if (read_word >= args.lanes || write_word >= args.lanes) {
        if (tid == 0u) {
            raise(faults, args.layer, BLOOMERY_FAULT_SITE_DELTA_LANE);
        }
        return;
    }

    const float sigma = dtv > 20.0f ? dtv : mamba2_log1p(exp(dtv));
    const float decay = exp(sigma * a_h);

    // Stage the conv outputs: item i < 128 is B channel i of the group,
    // 128 <= i < 256 its C channel i - 128, then the block's channels of x.
    // The ring slot of this token takes each channel's input once, from the
    // threadgroup that owns the channel.
    const uint ring_slot = pos % BLOOMERY_DELTA_RING_ROWS;
    const bool first_of_group = (h % per == 0u) && chunk == 0u;
    const uint c_x = h * hd + chunk * MAMBA2_CH_PER_BLOCK;
    for (uint round = 0u; round < 3u; ++round) {
        const uint item = tid + MAMBA2_THREADS * round;
        if (item < MAMBA2_STAGED) {
            uint c;
            bool live;
            bool writes;
            if (item < MAMBA2_STATE) {
                c = inner + g * MAMBA2_STATE + item;
                live = true;
                writes = first_of_group;
            } else if (item < 2u * MAMBA2_STATE) {
                c = inner + args.n_group * MAMBA2_STATE + g * MAMBA2_STATE + (item - MAMBA2_STATE);
                live = true;
                writes = first_of_group;
            } else {
                const uint k = item - 2u * MAMBA2_STATE;
                c = c_x + k;
                live = mamba2_live(chunk, k, hd);
                writes = true;
            }
            if (live) {
                const float cur = xbc[c];
                const float x0 = mamba2_tap(ring, pos, 3u, conv_dim, c);
                const float x1 = mamba2_tap(ring, pos, 2u, conv_dim, c);
                const float x2 = mamba2_tap(ring, pos, 1u, conv_dim, c);
                stage[item] = mamba2_conv(x0, x1, x2, cur, conv_w, conv_b, c);
                if (writes) {
                    ring[ring_slot * conv_dim + c] = cur;
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float bq[4];
    float cq[4];
    for (uint j = 0u; j < 4u; ++j) {
        bq[j] = stage[lane + 32u * j];
        cq[j] = stage[MAMBA2_STATE + lane + 32u * j];
    }

    // The SIMD group's four channels: their state first, all loads in flight.
    float s[16];
    for (uint i = 0u; i < MAMBA2_CH_PER_SIMD_GROUP; ++i) {
        const uint k = sg * MAMBA2_CH_PER_SIMD_GROUP + i;
        if (mamba2_live(chunk, k, hd)) {
            const uint base = (read_word * inner + h * hd + chunk * MAMBA2_CH_PER_BLOCK + k) * MAMBA2_STATE + lane;
            for (uint j = 0u; j < 4u; ++j) {
                s[4u * i + j] = state[base + 32u * j];
            }
        }
    }
    for (uint i = 0u; i < MAMBA2_CH_PER_SIMD_GROUP; ++i) {
        const uint k = sg * MAMBA2_CH_PER_SIMD_GROUP + i;
        if (mamba2_live(chunk, k, hd)) {
            const uint at = h * hd + chunk * MAMBA2_CH_PER_BLOCK + k;
            const float x = stage[2u * MAMBA2_STATE + k];
            const float x_dt = x * sigma;
            float acc = 0.0f;
            for (uint j = 0u; j < 4u; ++j) {
                const float u = s[4u * i + j] * decay + bq[j] * x_dt;
                s[4u * i + j] = u;
                acc += u * cq[j];
            }
            acc = butterfly_sum(acc);
            const float v = mamba2_silu(z[at]) * (acc + x * d_h);
            if (isfinite(v)) {
                const uint base = (write_word * inner + at) * MAMBA2_STATE + lane;
                for (uint j = 0u; j < 4u; ++j) {
                    state[base + 32u * j] = s[4u * i + j];
                }
            } else if (lane == 0u) {
                raise(faults, args.layer, BLOOMERY_FAULT_SITE_MAMBA2_SCAN);
            }
            if (lane == 0u) {
                y[at] = v;
                ys[k] = v;
            }
        }
    }

    // The block's sum of squares, in channel order.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float sum = 0.0f;
        for (uint k = 0u; k < MAMBA2_CH_PER_BLOCK; ++k) {
            if (mamba2_live(chunk, k, hd)) {
                sum += ys[k] * ys[k];
            }
        }
        part[tg] = sum;
    }
}

// One threadgroup a group: the group's partials in threadgroup order, the
// scale 1 / sqrt(mean + eps), and gain · scale · y over the group's channels.
kernel void mamba2_norm(device const float *y [[buffer(0)]],
                        device const float *part [[buffer(1)]],
                        device const float *gain [[buffer(2)]],
                        device float *out [[buffer(3)]],
                        constant Mamba2NormArgs &args [[buffer(4)]],
                        device atomic_uint *faults [[buffer(5)]],
                        uint g [[threadgroup_position_in_grid]],
                        uint tid [[thread_position_in_threadgroup]]) {
    threadgroup float scale_tg[1];
    threadgroup uint usable_tg[1];

    const uint per = args.n_head / args.n_group;
    const uint span = per * args.head_dim;
    // The heads whose partials the group's mean covers.
    const uint stat_heads = per;
    const uint stat_first = g * stat_heads * args.chunks;
    const uint stat_width = stat_heads * args.head_dim;
    if (tid == 0u) {
        float total = 0.0f;
        for (uint i = 0u; i < stat_heads * args.chunks; ++i) {
            total += part[stat_first + i];
        }
        const float scale = rms_scale(total, stat_width, args.eps);
        scale_tg[0] = scale;
        usable_tg[0] = rms_usable(total, scale) ? 1u : 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float scale = scale_tg[0];
    if (usable_tg[0] == 0u) {
        if (tid == 0u) {
            raise(faults, args.layer, BLOOMERY_FAULT_SITE_MAMBA2_SCAN);
        }
        for (uint i = tid; i < span; i += MAMBA2_THREADS) {
            out[g * span + i] = NAN;
        }
        return;
    }
    bool bad = false;
    for (uint i = tid; i < span; i += MAMBA2_THREADS) {
        const uint at = g * span + i;
        const float o = (scale * gain[at]) * y[at];
        bad = bad || !isfinite(o);
        out[at] = o;
    }
    if (bad) {
        raise(faults, args.layer, BLOOMERY_FAULT_SITE_MAMBA2_SCAN);
    }
}
