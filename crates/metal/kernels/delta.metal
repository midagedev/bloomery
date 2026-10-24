// The gated delta step of one decode token: causal conv and SiLU, the q/k
// L2 norm, β and the per-head decay, the delta update, the gated numerator
// and the partial sums of o². The norm scale is not applied here; the
// head-normed mat-vec multiplies it on load.
//
// The grid is 256 threadgroups of 128 threads: 32 value heads, 8 column
// blocks, 4 simdgroups. A simdgroup holds 4 columns of 8 lanes. Lane j of a
// column holds keys 32·r + 4·j + c. The q/k conv is redone in every
// threadgroup (256 channels, 4 taps).
//
// The ring stores the conv input. A predecessor at distance d is 0 while
// pos < d, otherwise the slot (pos − d) mod the ring length. The current token is the
// launch's x, not a ring slot. One threadgroup writes each channel: a q/k
// channel is the lowest value head of its key head and column block 0, a
// value channel is the threadgroup that owns the column. The written slot is
// not a slot this token reads.
//
// The rows launch (gdn_rows) runs the step's body, gdn_row, once for each of
// m rows of a chunk inside one launch, rows in order, the state held in
// registers between them: it reads the lane the word at `lane_at` names
// before the first row and writes the lane the word at `lane_to` names after
// the last. A tap that falls inside the chunk reads the chunk's own conv input
// row; one before it reads the ring. Every row is the bits of the stepped
// launch at its position. The ring is committed after the launch by
// gdn_ring_rows: the last min(m, ring) positions land in their slots, and the
// launch's own reads of earlier slots have finished by then.
//
// The state is read from the lane the word at `lane_at` names and written to
// the lane the word at `lane_to` names. The two words may name one lane: the
// step then updates it in place. Two lanes apart, the step leaves the lane it
// read as it was.
//
// A lane word at or past the lane count, read or written, raises the lane
// site and does not write the state. A lane whose state is not finite raises
// the same site and writes NaN on that lane only; the other lanes of its
// column write the state they loaded. A conv, a norm, β or a decay that is
// not finite raises the conv site and does not write the state. A non-finite
// output that is none of those raises the delta site. Nothing here raises the
// gate site.

#include <metal_stdlib>

using namespace metal;

#pragma METAL fp contract(off)

// The step's argument block, bound by value. Eight words: the epsilon is a float.
struct DeltaArgs {
    uint n_k;
    uint grouped; // 0 tiled (h % n_k), else grouped (h / (n_v / n_k))
    uint lanes;
    uint lane_at; // the word of the lane the step reads
    uint lane_to; // the word of the lane the step writes
    uint layer;
    uint pos;
    float eps;
};

static_assert(sizeof(DeltaArgs) == 32, "DeltaArgs is 32 bytes: the host packs eight words");

// The rows launch's argument block: the step's words, with `pos` the first
// row's position, then the row count and the strides between rows (in floats)
// of the conv input, the gate input, β, α, the numerator and the partials.
struct DeltaRowsArgs {
    DeltaArgs step;
    uint rows;
    uint x_stride;
    uint z_stride;
    uint b_stride;
    uint a_stride;
    uint g_stride;
    uint partials_stride;
};

static_assert(sizeof(DeltaRowsArgs) == 60, "DeltaRowsArgs is 60 bytes: the host packs fifteen words");

// The ring commit's argument block: the channels, the rows of the chunk, the
// conv input's row stride and the first row's position.
struct DeltaRingArgs {
    uint ch;
    uint rows;
    uint x_stride;
    uint pos;
};

static_assert(sizeof(DeltaRingArgs) == 16, "DeltaRingArgs is 16 bytes: the host packs four words");
static_assert(BLOOMERY_DELTA_HEAD_COLS / BLOOMERY_DELTA_COLUMN_BLOCKS == BLOOMERY_DELTA_COLS_PER_BLOCK,
              "a column block is the head divided by the block count");

constant float Q_SCALE = as_type<float>(0x3db504f3u);

inline uint key_of(uint j, uint i) {
    return 32u * (i / 4u) + 4u * j + (i % 4u);
}

// The conv input d steps before row r of a chunk that starts at position
// pos0. Distance 0 is this row's x. A predecessor inside the chunk (d <= r)
// is the chunk's own row r - d; one before it is the ring slot of its
// position, or 0 while that position is before the first.
inline float tap_x(uint c, uint pos0, uint r, uint d, uint ch, uint x_stride,
                   device const float *x, device const float *ring) {
    if (d <= r) {
        return x[(r - d) * x_stride + c];
    }
    uint back = d - r;
    if (pos0 < back) {
        return 0.0f;
    }
    uint s = (pos0 + BLOOMERY_DELTA_RING_ROWS - back) % BLOOMERY_DELTA_RING_ROWS;
    return ring[s * ch + c];
}

// SiLU(conv4) of channel c, and whether every value stayed finite.
inline float conv_silu(uint c, uint pos0, uint r, uint ch, uint x_stride,
                       device const float *x, device const float *ring,
                       device const float *w, thread bool *ok) {
    float x0 = tap_x(c, pos0, r, 3u, ch, x_stride, x, ring);
    float x1 = tap_x(c, pos0, r, 2u, ch, x_stride, x, ring);
    float x2 = tap_x(c, pos0, r, 1u, ch, x_stride, x, ring);
    float x3 = tap_x(c, pos0, r, 0u, ch, x_stride, x, ring);
    float w0 = w[4u * c];
    float w1 = w[4u * c + 1u];
    float w2 = w[4u * c + 2u];
    float w3 = w[4u * c + 3u];
    float acc = x0 * w0;
    acc = fma(x1, w1, acc);
    acc = fma(x2, w2, acc);
    acc = fma(x3, w3, acc);
    float y = acc / (1.0f + exp(-acc));
    if (!isfinite(x0) || !isfinite(x1) || !isfinite(x2) || !isfinite(x3) ||
        !isfinite(w0) || !isfinite(w1) || !isfinite(w2) || !isfinite(w3) ||
        !isfinite(acc) || !isfinite(y)) {
        *ok = false;
    }
    return y;
}

inline float softplus(float x) {
    if (x > 20.0f) {
        return x;
    }
    return log(1.0f + exp(x));
}

inline float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

// One lane's sixteen products, in the contract's (r, c) order.
inline float lane_dot(thread const float *a, thread const float *b) {
    float p0 = a[0] * b[0];
    float p1 = a[1] * b[1];
    float p2 = a[2] * b[2];
    float p3 = a[3] * b[3];
    for (uint r = 1u; r < 4u; ++r) {
        p0 = fma(a[4u * r], b[4u * r], p0);
        p1 = fma(a[4u * r + 1u], b[4u * r + 1u], p1);
        p2 = fma(a[4u * r + 2u], b[4u * r + 2u], p2);
        p3 = fma(a[4u * r + 3u], b[4u * r + 3u], p3);
    }
    return (p0 + p1) + (p2 + p3);
}

// The eight lanes of one column. Distances 4, 2, 1 stay inside the column:
// the eight lanes are eight-aligned.
inline float column_sum(float x) {
    x += simd_shuffle_xor(x, ushort(4));
    x += simd_shuffle_xor(x, ushort(2));
    x += simd_shuffle_xor(x, ushort(1));
    return x;
}

inline uint column_or(uint x) {
    x |= simd_shuffle_xor(x, ushort(4));
    x |= simd_shuffle_xor(x, ushort(2));
    x |= simd_shuffle_xor(x, ushort(1));
    return x;
}

// The whole simdgroup, in the order the delta's sum contract names:
// distances 16, 8, 4, 2, 1. These are not the language's reductions.
inline uint butterfly_or(uint x) {
    x |= simd_shuffle_xor(x, ushort(16));
    x |= simd_shuffle_xor(x, ushort(8));
    x |= simd_shuffle_xor(x, ushort(4));
    x |= simd_shuffle_xor(x, ushort(2));
    x |= simd_shuffle_xor(x, ushort(1));
    return x;
}

inline float butterfly_sum(float x) {
    x += simd_shuffle_xor(x, ushort(16));
    x += simd_shuffle_xor(x, ushort(8));
    x += simd_shuffle_xor(x, ushort(4));
    x += simd_shuffle_xor(x, ushort(2));
    x += simd_shuffle_xor(x, ushort(1));
    return x;
}

// 1 / max(sqrt(sum), eps). A non-finite sum is NaN, not max(NaN, eps).
inline float l2_scale(float sum, float eps) {
    if (!isfinite(sum)) {
        return NAN;
    }
    float root = sqrt(sum);
    float denom = root > eps ? root : eps;
    return 1.0f / denom;
}

// Where a thread sits in the step, the same for every row of a launch.
struct GdnSeat {
    uint tid;
    uint n_k;
    uint ch;
    uint h;
    uint block;
    uint j;
    uint col;
    uint col_in_block;
    uint kh;
    bool qk_owner;
    ushort tiisg;
    ushort sgitg;
};

inline GdnSeat gdn_seat(uint3 tgp, uint3 tpt, ushort tiisg, ushort sgitg, uint n_k,
                        uint grouped) {
    GdnSeat s;
    uint n_v = BLOOMERY_DELTA_VALUE_HEADS;
    uint head = BLOOMERY_DELTA_HEAD_COLS;
    uint tg = tgp.x;
    s.tid = tpt.x;
    s.n_k = n_k;
    s.ch = (2u * n_k + n_v) * head;
    s.block = tg % BLOOMERY_DELTA_COLUMN_BLOCKS;
    s.h = tg / BLOOMERY_DELTA_COLUMN_BLOCKS;
    s.j = (uint)(tiisg % 8);
    uint sub = (uint)(tiisg / 8);
    s.col = s.block * BLOOMERY_DELTA_COLS_PER_BLOCK + (uint)sgitg * 4u + sub;
    s.col_in_block = (uint)sgitg * 4u + sub;
    uint per = n_v / n_k;
    s.kh = grouped != 0u ? s.h / per : s.h % n_k;
    bool lowest = grouped != 0u ? (s.h % per == 0u) : (s.h < n_k);
    s.qk_owner = s.block == 0u && lowest;
    s.tiisg = tiisg;
    s.sgitg = sgitg;
    return s;
}

// The buffers every row of a launch reads, and the words that name the layer.
struct GdnBound {
    device const float *x;
    device const float *ring;
    device const float *conv_w;
    device const float *dt_bias;
    device const float *ssm_a;
    device const float *norm_w;
    device atomic_uint *faults;
    uint x_stride;
    uint layer;
    float eps;
};

// The threadgroup memory one row works in, reused by the next row.
struct GdnShared {
    threadgroup float *qv;
    threadgroup float *kv;
    threadgroup float *o2;
    threadgroup uint *sg_bad;
    threadgroup uint *scale_bad;
};

// One row of the step: the conv and SiLU, the q/k norm, β and the decay, the
// delta update and the gated numerator. `st` is the lane's sixteen state
// values: the state the row reads, and after the call the state the step
// leaves in the lane (a row that writes no state leaves them as they were).
// The result is whether the step writes the lane: false when the lane word
// was out of range or the conv was not usable. Row r sits at position
// pos0 + r; its conv input is row r of `b.x`, its predecessors before r come
// from the chunk and the earlier ones from the ring. This body is the decode
// step's and every row kernel's: the rows are bit-equal the stepped launches.
// The row ends past the barrier that closes its reduction, so the next row
// may reuse the shared memory.
inline bool gdn_row(thread const GdnSeat &s, thread const GdnBound &b, uint pos0, uint r,
                    bool lane_ok, device const float *b_row, device const float *a_row,
                    device const float *z_row, device float *g_row, device float *partials_row,
                    thread float *st, thread const GdnShared &sh) {
    uint tid = s.tid;
    uint head = BLOOMERY_DELTA_HEAD_COLS;
    uint ch = s.ch;
    uint h = s.h;
    uint j = s.j;
    uint col = s.col;
    uint kh = s.kh;
    uint n_k = s.n_k;
    ushort tiisg = s.tiisg;
    ushort sgitg = s.sgitg;

    if (tid == 0u) {
        *sh.scale_bad = 0u;
    }

    bool q_ok = true;
    bool k_ok = true;
    float qs = conv_silu(kh * head + tid, pos0, r, ch, b.x_stride, b.x, b.ring, b.conv_w, &q_ok);
    float ks = conv_silu((n_k + kh) * head + tid, pos0, r, ch, b.x_stride, b.x, b.ring, b.conv_w,
                         &k_ok);
    sh.qv[tid] = qs;
    sh.kv[tid] = ks;

    float v = 0.0f;
    bool v_ok = true;
    if (j == 0u) {
        uint vc = (2u * n_k + h) * head + col;
        v = conv_silu(vc, pos0, r, ch, b.x_stride, b.x, b.ring, b.conv_w, &v_ok);
    }
    v = simd_shuffle(v, (ushort)((tiisg / 8) * 8));
    // v_ok is the j == 0 thread's. Spread it with the flag below; a shuffle
    // of the value already gave every lane of the column the same v.

    float ba = b_row[h];
    float aa = a_row[h];
    float dt = b.dt_bias[h];
    float sa = b.ssm_a[h];
    float decay = exp(sa * softplus(aa + dt));
    float beta = sigmoid(ba);
    bool prep_ok = isfinite(ba) && isfinite(aa) && isfinite(dt) && isfinite(sa) &&
                   isfinite(decay) && isfinite(beta) && isfinite(v);
    uint my_bad = (q_ok && k_ok && v_ok && prep_ok) ? 0u : 1u;
    // j != 0 did not run the value conv, so its v_ok is still true. The j == 0
    // thread's flag reaches the whole simdgroup through this or.
    my_bad = butterfly_or(my_bad);
    if (tiisg == 0u) {
        sh.sg_bad[sgitg] = my_bad;
    }

    // q and k are in threadgroup memory; the norm reads them.
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0u) {
        float q0 = sh.qv[tiisg];
        float q1 = sh.qv[tiisg + 32u];
        float q2 = sh.qv[tiisg + 64u];
        float q3 = sh.qv[tiisg + 96u];
        float k0 = sh.kv[tiisg];
        float k1 = sh.kv[tiisg + 32u];
        float k2 = sh.kv[tiisg + 64u];
        float k3 = sh.kv[tiisg + 96u];
        float sumq = (q0 * q0 + q1 * q1) + (q2 * q2 + q3 * q3);
        float sumk = (k0 * k0 + k1 * k1) + (k2 * k2 + k3 * k3);
        sumq = butterfly_sum(sumq);
        sumk = butterfly_sum(sumk);
        float sq = l2_scale(sumq, b.eps);
        float sk = l2_scale(sumk, b.eps);
        if (tiisg == 0u && (!isfinite(sq) || !isfinite(sk))) {
            *sh.scale_bad = 1u;
        }
        sh.qv[tiisg] = (q0 * sq) * Q_SCALE;
        sh.qv[tiisg + 32u] = (q1 * sq) * Q_SCALE;
        sh.qv[tiisg + 64u] = (q2 * sq) * Q_SCALE;
        sh.qv[tiisg + 96u] = (q3 * sq) * Q_SCALE;
        sh.kv[tiisg] = k0 * sk;
        sh.kv[tiisg + 32u] = k1 * sk;
        sh.kv[tiisg + 64u] = k2 * sk;
        sh.kv[tiisg + 96u] = k3 * sk;
    }

    // Scaled q and k are in threadgroup memory; every column reads them.
    threadgroup_barrier(mem_flags::mem_threadgroup);

    bool conv_bad = (sh.sg_bad[0] | sh.sg_bad[1] | sh.sg_bad[2] | sh.sg_bad[3] | *sh.scale_bad) != 0u;

    float sp[16];
    float kk[16];
    float qq[16];
    bool lane_state_bad = false;
    if (lane_ok) {
        for (uint i = 0u; i < 16u; ++i) {
            if (!isfinite(st[i])) {
                lane_state_bad = true;
            }
        }
    }
    for (uint i = 0u; i < 16u; ++i) {
        uint key = key_of(j, i);
        kk[i] = sh.kv[key];
        qq[i] = sh.qv[key];
        sp[i] = (lane_ok && !lane_state_bad) ? st[i] * decay : 0.0f;
    }

    float kvdot = column_sum(lane_dot(sp, kk));
    float u = (v - kvdot) * beta;
    for (uint i = 0u; i < 16u; ++i) {
        sp[i] = fma(kk[i], u, sp[i]);
    }
    float o = column_sum(lane_dot(sp, qq));

    uint lb = lane_state_bad ? 1u : 0u;
    lb = column_or(lb);
    bool column_bad = lane_ok && !conv_bad && lb != 0u;

    float gval = NAN;
    float oval = NAN;
    if (lane_ok && !conv_bad && !column_bad) {
        oval = o;
        float zv = z_row[h * head + col];
        gval = (o * b.norm_w[col]) * (zv / (1.0f + exp(-zv)));
    }
    if (j == 0u) {
        sh.o2[s.col_in_block] = oval * oval;
        g_row[h * head + col] = gval;
    }

    // One square per column of this block; thread 0 sums them. The third
    // barrier: the norm's two do not cover this reduction.
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        float p = sh.o2[0];
        for (uint i = 1u; i < 16u; ++i) {
            p += sh.o2[i];
        }
        if (!lane_ok || conv_bad) {
            p = NAN;
        }
        partials_row[h * BLOOMERY_DELTA_COLUMN_BLOCKS + s.block] = p;
    }

    if (!lane_ok) {
        if (tid == 0u) {
            raise(b.faults, b.layer, BLOOMERY_FAULT_SITE_DELTA_LANE);
        }
        return false;
    }
    if (conv_bad) {
        if (tid == 0u) {
            raise(b.faults, b.layer, BLOOMERY_FAULT_SITE_LINEAR_CONV);
        }
        return false;
    }
    if (column_bad) {
        if (lane_state_bad) {
            for (uint i = 0u; i < 16u; ++i) {
                st[i] = NAN;
            }
            raise(b.faults, b.layer, BLOOMERY_FAULT_SITE_DELTA_LANE);
        }
        return true;
    }
    for (uint i = 0u; i < 16u; ++i) {
        st[i] = sp[i];
    }
    if (j == 0u && !isfinite(o)) {
        raise(b.faults, b.layer, BLOOMERY_FAULT_SITE_LINEAR_DELTA);
    }
    return true;
}

// The lane's sixteen state values at `base`, and whether the lane word was in
// range. A lane word out of range leaves them zero.
inline void gdn_load(device const float *state, uint base, uint j, bool lane_ok,
                     thread float *st) {
    for (uint i = 0u; i < 16u; ++i) {
        st[i] = 0.0f;
    }
    if (lane_ok) {
        for (uint i = 0u; i < 16u; ++i) {
            st[i] = state[base + key_of(j, i)];
        }
    }
}

kernel void gdn_step(device const float *x [[buffer(0)]],
                     device const float *conv_w [[buffer(1)]],
                     device const float *b_raw [[buffer(2)]],
                     device const float *a_raw [[buffer(3)]],
                     device const float *dt_bias [[buffer(4)]],
                     device const float *ssm_a [[buffer(5)]],
                     device const float *z [[buffer(6)]],
                     device const float *norm_w [[buffer(7)]],
                     device const uint *lane_buf [[buffer(8)]],
                     device float *state [[buffer(9)]],
                     device float *ring [[buffer(10)]],
                     device float *partials [[buffer(11)]],
                     device float *g [[buffer(12)]],
                     constant DeltaArgs &args [[buffer(13)]],
                     device atomic_uint *faults [[buffer(14)]],
                     uint3 tgp [[threadgroup_position_in_grid]],
                     uint3 tpt [[thread_position_in_threadgroup]],
                     ushort tiisg [[thread_index_in_simdgroup]],
                     ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    GdnSeat seat = gdn_seat(tgp, tpt, tiisg, sgitg, args.n_k, args.grouped);
    uint n_v = BLOOMERY_DELTA_VALUE_HEADS;
    uint head = BLOOMERY_DELTA_HEAD_COLS;
    uint ch = seat.ch;
    uint pos = args.pos;

    threadgroup float qv[128];
    threadgroup float kv[128];
    threadgroup float o2[16];
    threadgroup uint sg_bad[4];
    threadgroup uint scale_bad;
    GdnShared sh = {qv, kv, o2, sg_bad, &scale_bad};
    GdnBound bound = {x, ring, conv_w, dt_bias, ssm_a, norm_w, faults, 0u, args.layer, args.eps};

    uint word = lane_buf[args.lane_at];
    uint word_to = lane_buf[args.lane_to];
    bool lane_ok = word < args.lanes && word_to < args.lanes;
    uint base = ((word * n_v + seat.h) * head + seat.col) * head;
    uint base_to = ((word_to * n_v + seat.h) * head + seat.col) * head;

    float st[16];
    gdn_load(state, base, seat.j, lane_ok, st);
    bool wrote = gdn_row(seat, bound, pos, 0u, lane_ok, b_raw, a_raw, z, g, partials, st, sh);
    if (wrote) {
        for (uint i = 0u; i < 16u; ++i) {
            state[base_to + key_of(seat.j, i)] = st[i];
        }
    }

    // The conv input of this token lands in its ring slot, once per channel.
    // No threadgroup reads that slot in this launch.
    uint slot = pos % BLOOMERY_DELTA_RING_ROWS;
    if (seat.qk_owner) {
        uint qc = seat.kh * head + seat.tid;
        uint kc = (seat.n_k + seat.kh) * head + seat.tid;
        ring[slot * ch + qc] = x[qc];
        ring[slot * ch + kc] = x[kc];
    }
    if (seat.j == 0u) {
        uint vc = (2u * seat.n_k + seat.h) * head + seat.col;
        ring[slot * ch + vc] = x[vc];
    }
}

// The step over a chunk of rows in one launch: the rows in order, the state
// read once from the lane the word at `lane_at` names and written once, after
// the last row, to the lane the word at `lane_to` names. The state stays in
// the lane's registers between rows, so m rows leave the bits m stepped
// launches leave. Rows are strided buffers; the chunk starts at `step.pos`.
// The ring is only read here: a threadgroup's later rows would write slots
// that another threadgroup's first rows still read, so gdn_ring_rows commits
// the ring in a launch of its own.
kernel void gdn_rows(device const float *x [[buffer(0)]],
                     device const float *conv_w [[buffer(1)]],
                     device const float *b_raw [[buffer(2)]],
                     device const float *a_raw [[buffer(3)]],
                     device const float *dt_bias [[buffer(4)]],
                     device const float *ssm_a [[buffer(5)]],
                     device const float *z [[buffer(6)]],
                     device const float *norm_w [[buffer(7)]],
                     device const uint *lane_buf [[buffer(8)]],
                     device float *state [[buffer(9)]],
                     device const float *ring [[buffer(10)]],
                     device float *partials [[buffer(11)]],
                     device float *g [[buffer(12)]],
                     constant DeltaRowsArgs &args [[buffer(13)]],
                     device atomic_uint *faults [[buffer(14)]],
                     uint3 tgp [[threadgroup_position_in_grid]],
                     uint3 tpt [[thread_position_in_threadgroup]],
                     ushort tiisg [[thread_index_in_simdgroup]],
                     ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    GdnSeat seat = gdn_seat(tgp, tpt, tiisg, sgitg, args.step.n_k, args.step.grouped);
    uint n_v = BLOOMERY_DELTA_VALUE_HEADS;
    uint head = BLOOMERY_DELTA_HEAD_COLS;

    threadgroup float qv[128];
    threadgroup float kv[128];
    threadgroup float o2[16];
    threadgroup uint sg_bad[4];
    threadgroup uint scale_bad;
    GdnShared sh = {qv, kv, o2, sg_bad, &scale_bad};
    GdnBound bound = {x,       ring,       conv_w,         dt_bias,        ssm_a,   norm_w,
                      faults,  args.x_stride, args.step.layer, args.step.eps};

    uint word = lane_buf[args.step.lane_at];
    uint word_to = lane_buf[args.step.lane_to];
    bool lane_ok = word < args.step.lanes && word_to < args.step.lanes;
    uint base = ((word * n_v + seat.h) * head + seat.col) * head;
    uint base_to = ((word_to * n_v + seat.h) * head + seat.col) * head;

    float st[16];
    gdn_load(state, base, seat.j, lane_ok, st);
    for (uint r = 0u; r < args.rows; ++r) {
        gdn_row(seat, bound, args.step.pos, r, lane_ok, b_raw + r * args.b_stride,
                a_raw + r * args.a_stride, z + r * args.z_stride, g + r * args.g_stride,
                partials + r * args.partials_stride, st, sh);
    }
    if (lane_ok) {
        for (uint i = 0u; i < 16u; ++i) {
            state[base_to + key_of(seat.j, i)] = st[i];
        }
    }
}

// The ring after a chunk of rows: the slots of the last min(rows, ring)
// positions hold the conv inputs of those rows, the others stay. Grid: the
// channels across, one row of the grid for each kept position.
kernel void gdn_ring_rows(device const float *x [[buffer(0)]],
                          device float *ring [[buffer(1)]],
                          constant DeltaRingArgs &args [[buffer(2)]],
                          uint2 gid [[thread_position_in_grid]]) {
    uint c = gid.x;
    uint i = gid.y;
    uint kept = min(args.rows, (uint)BLOOMERY_DELTA_RING_ROWS);
    if (c >= args.ch || i >= kept) {
        return;
    }
    uint r = args.rows - kept + i;
    uint slot = (args.pos + r) % BLOOMERY_DELTA_RING_ROWS;
    ring[slot * args.ch + c] = x[r * args.x_stride + c];
}
