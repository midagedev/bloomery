// The gated delta step of one decode token: causal conv and SiLU, the q/k
// L2 norm, β and the per-head decay, the delta update, the gated numerator
// and the partial sums of o². The norm scale is not applied here; the
// head-normed mat-vec multiplies it on load.
//
// A row has two halves. The state-free half is everything a row computes
// from its own projections and the conv ring: the conv and SiLU of q, k and v,
// the q/k norm, β, the decay and the gate's SiLU. The recurrence half is the
// delta update of the state, the column outputs and the state write. gdn_step
// runs both halves inside one launch (gdn_row). The rows launch runs the
// state-free half of every row of a chunk as one parallel launch
// (gdn_prep_rows), the recurrence in rows order (gdn_rows) and then the
// numerators and partial sums as one parallel launch (gdn_post_rows). Each
// function of a half is one inline function that both launches call, so a row
// is the bits of the stepped launch at its position.
//
// The step's grid is one threadgroup of 128 threads for each value head and
// column block: BLOOMERY_DELTA_VALUE_HEADS × 8 of them, 4 simdgroups each. A
// simdgroup holds 4 columns of 8 lanes. Lane j of a column holds keys
// 32·r + 4·j + c. The q/k conv is redone in every threadgroup (256 channels,
// 4 taps). A key head serves n_v / n_k value heads, the count the argument
// block's n_k sets at run time.
//
// The ring stores the conv input. A predecessor at distance d is 0 while
// pos < d, otherwise the slot (pos − d) mod the ring length. The current token is the
// launch's x, not a ring slot. One threadgroup writes each channel: a q/k
// channel is the lowest value head of its key head and column block 0, a
// value channel is the threadgroup that owns the column. The written slot is
// not a slot this token reads.
//
// The rows launch. gdn_prep_rows is a grid of (key head, row) threadgroups of
// 128 threads: thread t is channel t of the key head's q and k, and column t
// of each value head that reads the key head. A tap that falls inside the
// chunk reads the chunk's own conv input row; one before it reads the ring.
// It writes the prepared row (the layout the host names in GdnPrepLayout):
// the normed and scaled q and k, the value conv outputs, the gate's SiLU, β
// and the decay, and a status word for each recurrence threadgroup that is
// non-zero when a conv, a norm, β or a decay of that group's row is not
// finite. gdn_rows is the step's threadgroups with the state in
// registers across the rows: it reads the lane the word at `lane_at` names
// before the first row and writes the lane the word at `lane_to` names after
// the last, and a row reads only its prepared row. It writes the column
// outputs o into the numerator window, NaN where the row is unusable.
// gdn_post_rows turns them into the numerators and the partial sums of o². The
// ring is committed by gdn_ring_rows: the last min(m, ring) positions land in
// their slots, after the prepared rows were read.
//
// A rows launch may also leave copies after rows inside the chunk (a prompt
// pass's marks): gdn_rows_snap is gdn_rows that writes the lane's registers
// after each named row into that copy's lane, and gdn_ring_snap writes the
// ring a chunk ending after that row would leave into that copy's ring, read
// before gdn_ring_rows overwrites the slots. A copy is the bits a chunk ending
// at its row leaves; the chunk's own outputs do not move.
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

// Where the parts of a prepared row sit, in floats from the row's start, and
// the row stride. The scaled q and k are the first two parts, the key heads'
// 128 values each, so q starts at 0. Every part starts on a 16-byte boundary.
struct GdnPrepLayout {
    uint stride;
    uint k;     // the scaled keys
    uint v;     // the value conv outputs, 128 a value head
    uint gate;  // the gate's SiLU, 128 a value head
    uint decay; // one a value head
    uint beta;  // one a value head
    uint bad;   // one status word a recurrence threadgroup
};

static_assert(sizeof(GdnPrepLayout) == 28, "GdnPrepLayout is 28 bytes: the host packs seven words");

// The rows launch's argument block: the step's words, with `pos` the first
// row's position, then the row count and the strides between rows (in floats)
// of the conv input, the gate input, β, α, the numerator and the partials, and
// the prepared rows' layout.
struct DeltaRowsArgs {
    DeltaArgs step;
    uint rows;
    uint x_stride;
    uint z_stride;
    uint b_stride;
    uint a_stride;
    uint g_stride;
    uint partials_stride;
    GdnPrepLayout prep;
};

static_assert(sizeof(DeltaRowsArgs) == 88, "DeltaRowsArgs is 88 bytes: the host packs twenty-two words");

// The ring commit's argument block: the channels, the rows of the chunk, the
// conv input's row stride and the first row's position.
struct DeltaRingArgs {
    uint ch;
    uint rows;
    uint x_stride;
    uint pos;
};

static_assert(sizeof(DeltaRingArgs) == 16, "DeltaRingArgs is 16 bytes: the host packs four words");

// The copies a rows launch leaves: how many, the floats of one copy's lane
// and of one copy's ring, and the row each copy follows, ascending.
struct DeltaSnapArgs {
    uint n;
    uint lane_len;
    uint ring_len;
    uint after[BLOOMERY_DELTA_SNAPS];
};

static_assert(sizeof(DeltaSnapArgs) == 4 * (3 + BLOOMERY_DELTA_SNAPS),
              "DeltaSnapArgs is three words and a row word a copy: the host packs them");
static_assert(BLOOMERY_DELTA_HEAD_COLS / BLOOMERY_DELTA_COLUMN_BLOCKS == BLOOMERY_DELTA_COLS_PER_BLOCK,
              "a column block is the head divided by the block count");
static_assert(BLOOMERY_DELTA_VALUE_HEADS % 2 == 0,
              "a prepared row's floats are a multiple of four only for an even value head count: "
              "the recurrence reads the keys four at a time");

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
    uint s = (pos0 - back) % BLOOMERY_DELTA_RING_ROWS;
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

// The buffers the decode step reads, and the words that name the layer.
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

// The threadgroup memory the decode step works in.
struct GdnShared {
    threadgroup float *qv;
    threadgroup float *kv;
    threadgroup float *o2;
    threadgroup uint *sg_bad;
    threadgroup uint *scale_bad;
};

// The row's functions take arrays of the lane's registers by pointer. They are
// forced inline so the arrays stay in registers: a call that is not inlined
// puts them in stack memory, which Metal backs with a scratch buffer.
#define GDN_INLINE inline __attribute__((__always_inline__))

// The functions of the state-free half, then of the recurrence half. The
// decode step and the rows launches call these, and there is no other copy of
// their math.

// SiLU of the gate input: the numerator's second factor.
GDN_INLINE float gdn_gate(float zv) {
    return zv / (1.0f + exp(-zv));
}

// β and the decay of a value head from its raw inputs, and whether every input
// and both results are finite.
GDN_INLINE bool gdn_scalars(float ba, float aa, float dt, float sa, thread float *decay,
                        thread float *beta) {
    *decay = exp(sa * softplus(aa + dt));
    *beta = sigmoid(ba);
    return isfinite(ba) && isfinite(aa) && isfinite(dt) && isfinite(sa) &&
           isfinite(*decay) && isfinite(*beta);
}

// The q/k norm of one key head, on its 128 conv outputs in threadgroup memory:
// one simdgroup scales both in place, q also by the query scale. The result is
// whether a norm scale was unusable, the same on every lane.
GDN_INLINE bool gdn_norm_qk(threadgroup float *qv, threadgroup float *kv, ushort tiisg, float eps) {
    float q0 = qv[tiisg];
    float q1 = qv[tiisg + 32u];
    float q2 = qv[tiisg + 64u];
    float q3 = qv[tiisg + 96u];
    float k0 = kv[tiisg];
    float k1 = kv[tiisg + 32u];
    float k2 = kv[tiisg + 64u];
    float k3 = kv[tiisg + 96u];
    float sumq = (q0 * q0 + q1 * q1) + (q2 * q2 + q3 * q3);
    float sumk = (k0 * k0 + k1 * k1) + (k2 * k2 + k3 * k3);
    sumq = butterfly_sum(sumq);
    sumk = butterfly_sum(sumk);
    float sq = l2_scale(sumq, eps);
    float sk = l2_scale(sumk, eps);
    qv[tiisg] = (q0 * sq) * Q_SCALE;
    qv[tiisg + 32u] = (q1 * sq) * Q_SCALE;
    qv[tiisg + 64u] = (q2 * sq) * Q_SCALE;
    qv[tiisg + 96u] = (q3 * sq) * Q_SCALE;
    kv[tiisg] = k0 * sk;
    kv[tiisg + 32u] = k1 * sk;
    kv[tiisg + 64u] = k2 * sk;
    kv[tiisg + 96u] = k3 * sk;
    return !isfinite(sq) || !isfinite(sk);
}

// The delta update of one lane's sixteen state values `st`, into `sp`, and the
// column's output: NaN when the lane word, the conv or the column is unusable.
// `lane_state_bad` is whether this lane's state is not finite, `column_bad`
// whether any lane of the column's is while the rest of the row is usable.
GDN_INLINE float gdn_delta(thread const GdnSeat &s, bool lane_ok, bool conv_bad,
                       thread const float *kk, thread const float *qq, float v, float decay,
                       float beta, thread const float *st, thread float *sp,
                       thread bool *lane_state_bad, thread bool *column_bad) {
    bool state_bad = false;
    if (lane_ok) {
        for (uint i = 0u; i < 16u; ++i) {
            if (!isfinite(st[i])) {
                state_bad = true;
            }
        }
    }
    for (uint i = 0u; i < 16u; ++i) {
        sp[i] = (lane_ok && !state_bad) ? st[i] * decay : 0.0f;
    }

    float kvdot = column_sum(lane_dot(sp, kk));
    float u = (v - kvdot) * beta;
    for (uint i = 0u; i < 16u; ++i) {
        sp[i] = fma(kk[i], u, sp[i]);
    }
    float o = column_sum(lane_dot(sp, qq));

    uint lb = state_bad ? 1u : 0u;
    lb = column_or(lb);
    bool bad = lane_ok && !conv_bad && lb != 0u;
    *lane_state_bad = state_bad;
    *column_bad = bad;
    return (lane_ok && !conv_bad && !bad) ? o : NAN;
}

// What a row leaves after its output: the raises, and the state `st` the step
// leaves in the lane. The result is whether the step writes the lane: false
// when the lane word was out of range or the conv was not usable.
GDN_INLINE bool gdn_settle(thread const GdnSeat &s, device atomic_uint *faults, uint layer,
                       bool lane_ok, bool conv_bad, bool column_bad, bool lane_state_bad,
                       float oval, thread const float *sp, thread float *st) {
    if (!lane_ok) {
        if (s.tid == 0u) {
            raise(faults, layer, BLOOMERY_FAULT_SITE_DELTA_LANE);
        }
        return false;
    }
    if (conv_bad) {
        if (s.tid == 0u) {
            raise(faults, layer, BLOOMERY_FAULT_SITE_LINEAR_CONV);
        }
        return false;
    }
    if (column_bad) {
        if (lane_state_bad) {
            for (uint i = 0u; i < 16u; ++i) {
                st[i] = NAN;
            }
            raise(faults, layer, BLOOMERY_FAULT_SITE_DELTA_LANE);
        }
        return true;
    }
    for (uint i = 0u; i < 16u; ++i) {
        st[i] = sp[i];
    }
    if (s.j == 0u && !isfinite(oval)) {
        raise(faults, layer, BLOOMERY_FAULT_SITE_LINEAR_DELTA);
    }
    return true;
}

// The numerator of a column from its output, the norm gain and the gate's
// SiLU. An unusable output is NaN.
GDN_INLINE float gdn_numerator(float oval, float gain, float gate) {
    return isnan(oval) ? NAN : (oval * gain) * gate;
}

// The partial sum of the 16 squared outputs of a column block, in column order.
GDN_INLINE float gdn_block_sum(threadgroup const float *o2) {
    float p = o2[0];
    for (uint i = 1u; i < 16u; ++i) {
        p += o2[i];
    }
    return p;
}

// The sixteen lanes of a column block, aligned: distances 8, 4, 2, 1.
GDN_INLINE uint block_or(uint x) {
    x |= simd_shuffle_xor(x, ushort(8));
    x |= simd_shuffle_xor(x, ushort(4));
    x |= simd_shuffle_xor(x, ushort(2));
    x |= simd_shuffle_xor(x, ushort(1));
    return x;
}

// One row of the decode step: the conv and SiLU, the q/k norm, β and the
// decay, the delta update and the gated numerator. `st` is the lane's sixteen
// state values: the state the row reads, and after the call the state the
// step leaves in the lane (a row that writes no state leaves them as they
// were). The result is whether the step writes the lane. Row r sits at
// position pos0 + r; its conv input is row r of `b.x`, its predecessors before
// r come from the chunk and the earlier ones from the ring. The row ends past
// the barrier that closes its reduction, so the next row may reuse the shared
// memory.
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

    float decay;
    float beta;
    bool scalars_ok = gdn_scalars(b_row[h], a_row[h], b.dt_bias[h], b.ssm_a[h], &decay, &beta);
    bool prep_ok = scalars_ok && isfinite(v);
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
        bool scale_unusable = gdn_norm_qk(sh.qv, sh.kv, tiisg, b.eps);
        if (tiisg == 0u && scale_unusable) {
            *sh.scale_bad = 1u;
        }
    }

    // Scaled q and k are in threadgroup memory; every column reads them.
    threadgroup_barrier(mem_flags::mem_threadgroup);

    bool conv_bad = (sh.sg_bad[0] | sh.sg_bad[1] | sh.sg_bad[2] | sh.sg_bad[3] | *sh.scale_bad) != 0u;

    float sp[16];
    float kk[16];
    float qq[16];
    for (uint i = 0u; i < 16u; ++i) {
        uint key = key_of(j, i);
        kk[i] = sh.kv[key];
        qq[i] = sh.qv[key];
    }
    bool lane_state_bad;
    bool column_bad;
    float oval = gdn_delta(s, lane_ok, conv_bad, kk, qq, v, decay, beta, st, sp, &lane_state_bad,
                           &column_bad);

    if (j == 0u) {
        sh.o2[s.col_in_block] = oval * oval;
        g_row[h * head + col] = gdn_numerator(oval, b.norm_w[col], gdn_gate(z_row[h * head + col]));
    }

    // One square per column of this block; thread 0 sums them. The third
    // barrier: the norm's two do not cover this reduction.
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        float p = gdn_block_sum(sh.o2);
        partials_row[h * BLOOMERY_DELTA_COLUMN_BLOCKS + s.block] = p;
    }

    return gdn_settle(s, b.faults, b.layer, lane_ok, conv_bad, column_bad, lane_state_bad, oval, sp,
                      st);
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

// The state-free half of a chunk of rows. Grid: (key heads, rows) threadgroups
// of 128 threads. Thread t holds channel t of the key head's q and k, and
// column t of each value head that reads the key head; it runs the decode
// step's conv, norm, β, decay and gate functions on them and writes the
// prepared row.
kernel void gdn_prep_rows(device const float *x [[buffer(0)]],
                          device const float *conv_w [[buffer(1)]],
                          device const float *b_raw [[buffer(2)]],
                          device const float *a_raw [[buffer(3)]],
                          device const float *dt_bias [[buffer(4)]],
                          device const float *ssm_a [[buffer(5)]],
                          device const float *z [[buffer(6)]],
                          device const float *ring [[buffer(7)]],
                          device float *prep [[buffer(8)]],
                          constant DeltaRowsArgs &args [[buffer(9)]],
                          uint3 tgp [[threadgroup_position_in_grid]],
                          uint3 tpt [[thread_position_in_threadgroup]],
                          ushort tiisg [[thread_index_in_simdgroup]],
                          ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    uint n_v = BLOOMERY_DELTA_VALUE_HEADS;
    uint head = BLOOMERY_DELTA_HEAD_COLS;
    uint n_k = args.step.n_k;
    uint kh = tgp.x;
    uint r = tgp.y;
    uint tid = tpt.x;
    uint ch = (2u * n_k + n_v) * head;
    uint per = n_v / n_k;
    device float *row = prep + (ulong)r * args.prep.stride;
    device uint *bad = (device uint *)(row + args.prep.bad);

    threadgroup float qv[128];
    threadgroup float kv[128];
    threadgroup uint sg_bad[4];
    threadgroup uint scale_bad;

    bool q_ok = true;
    bool k_ok = true;
    qv[tid] = conv_silu(kh * head + tid, args.step.pos, r, ch, args.x_stride, x, ring, conv_w,
                        &q_ok);
    kv[tid] = conv_silu((n_k + kh) * head + tid, args.step.pos, r, ch, args.x_stride, x, ring,
                        conv_w, &k_ok);
    uint qk_bad = butterfly_or((q_ok && k_ok) ? 0u : 1u);
    if (tiisg == 0u) {
        sg_bad[sgitg] = qk_bad;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0u) {
        bool unusable = gdn_norm_qk(qv, kv, tiisg, args.step.eps);
        if (tiisg == 0u) {
            scale_bad = unusable ? 1u : 0u;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    row[kh * head + tid] = qv[tid];
    row[args.prep.k + kh * head + tid] = kv[tid];
    uint group_bad = sg_bad[0] | sg_bad[1] | sg_bad[2] | sg_bad[3] | scale_bad;

    for (uint i = 0u; i < per; ++i) {
        uint h = args.step.grouped != 0u ? kh * per + i : kh + i * n_k;
        bool v_ok = true;
        float v = conv_silu((2u * n_k + h) * head + tid, args.step.pos, r, ch, args.x_stride, x,
                            ring, conv_w, &v_ok);
        row[args.prep.v + h * head + tid] = v;
        row[args.prep.gate + h * head + tid] = gdn_gate(z[(ulong)r * args.z_stride + h * head + tid]);
        float decay;
        float beta;
        bool scalars_ok = gdn_scalars(b_raw[(ulong)r * args.b_stride + h],
                                      a_raw[(ulong)r * args.a_stride + h], dt_bias[h], ssm_a[h],
                                      &decay, &beta);
        if (tid == 0u) {
            row[args.prep.decay + h] = decay;
            row[args.prep.beta + h] = beta;
        }
        uint column_bad = (v_ok && isfinite(v)) ? 0u : 1u;
        uint block_bad = block_or(column_bad);
        if (tid % BLOOMERY_DELTA_COLS_PER_BLOCK == 0u) {
            bool unusable = group_bad != 0u || block_bad != 0u || !scalars_ok;
            bad[h * BLOOMERY_DELTA_COLUMN_BLOCKS + tid / BLOOMERY_DELTA_COLS_PER_BLOCK] =
                unusable ? 1u : 0u;
        }
    }
}

// The recurrence over a chunk of rows in one launch: the rows in order, the
// state read once from the lane the word at `lane_at` names and written once,
// after the last row, to the lane the word at `lane_to` names. The state stays
// in the lane's registers between rows, so m rows leave the bits m stepped
// launches leave. A row reads only its prepared row, and writes the column
// outputs into the numerator window (gdn_post_rows finishes them). The
// simdgroups of a threadgroup share nothing, so there is no barrier. With
// `snaps`, the registers after each row it names land in that copy's lane of
// `snap`, as the lane write after the last row would land them: a copy is the
// state a chunk ending at its row leaves, and no output of the chunk moves.
inline void gdn_rows_run(device const float *prep, device const uint *lane_buf,
                         device float *state, device float *g, constant DeltaRowsArgs &args,
                         device atomic_uint *faults, device float *snap,
                         constant DeltaSnapArgs *snaps, uint3 tgp, uint3 tpt, ushort tiisg,
                         ushort sgitg) {
    GdnSeat seat = gdn_seat(tgp, tpt, tiisg, sgitg, args.step.n_k, args.step.grouped);
    uint n_v = BLOOMERY_DELTA_VALUE_HEADS;
    uint head = BLOOMERY_DELTA_HEAD_COLS;

    uint word = lane_buf[args.step.lane_at];
    uint word_to = lane_buf[args.step.lane_to];
    bool lane_ok = word < args.step.lanes && word_to < args.step.lanes;
    uint base = ((word * n_v + seat.h) * head + seat.col) * head;
    uint base_to = ((word_to * n_v + seat.h) * head + seat.col) * head;
    uint cell = (seat.h * head + seat.col) * head;
    uint copies = snaps == nullptr ? 0u : snaps->n;
    uint copy = 0u;

    float st[16];
    gdn_load(state, base, seat.j, lane_ok, st);
    bool row_ok = lane_ok;
    for (uint r = 0u; r < args.rows; ++r) {
        device const float *row = prep + (ulong)r * args.prep.stride;
        device const float *qp = row + seat.kh * head;
        device const float *kp = row + args.prep.k + seat.kh * head;
        float kk[16];
        float qq[16];
        for (uint c = 0u; c < 4u; ++c) {
            float4 kf = *((device const float4 *)(kp + 32u * c + 4u * seat.j));
            float4 qf = *((device const float4 *)(qp + 32u * c + 4u * seat.j));
            kk[4u * c] = kf.x;
            kk[4u * c + 1u] = kf.y;
            kk[4u * c + 2u] = kf.z;
            kk[4u * c + 3u] = kf.w;
            qq[4u * c] = qf.x;
            qq[4u * c + 1u] = qf.y;
            qq[4u * c + 2u] = qf.z;
            qq[4u * c + 3u] = qf.w;
        }
        float v = row[args.prep.v + seat.h * head + seat.col];
        float decay = row[args.prep.decay + seat.h];
        float beta = row[args.prep.beta + seat.h];
        bool conv_bad = ((device const uint *)(row + args.prep.bad))[tgp.x] != 0u;
        float sp[16];
        bool lane_state_bad;
        bool column_bad;
        float oval = gdn_delta(seat, row_ok, conv_bad, kk, qq, v, decay, beta, st, sp,
                               &lane_state_bad, &column_bad);
        if (seat.j == 0u) {
            g[(ulong)r * args.g_stride + seat.h * head + seat.col] = oval;
        }
        gdn_settle(seat, faults, args.step.layer, row_ok, conv_bad, column_bad, lane_state_bad,
                   oval, sp, st);
        if (copy < copies && r == snaps->after[copy]) {
            if (lane_ok) {
                device float *lane = snap + (ulong)copy * snaps->lane_len + cell;
                for (uint i = 0u; i < 16u; ++i) {
                    lane[key_of(seat.j, i)] = st[i];
                }
            }
            ++copy;
        }
    }
    if (lane_ok) {
        for (uint i = 0u; i < 16u; ++i) {
            state[base_to + key_of(seat.j, i)] = st[i];
        }
    }
}

kernel void gdn_rows(device const float *prep [[buffer(0)]],
                     device const uint *lane_buf [[buffer(1)]],
                     device float *state [[buffer(2)]],
                     device float *g [[buffer(3)]],
                     constant DeltaRowsArgs &args [[buffer(4)]],
                     device atomic_uint *faults [[buffer(5)]],
                     uint3 tgp [[threadgroup_position_in_grid]],
                     uint3 tpt [[thread_position_in_threadgroup]],
                     ushort tiisg [[thread_index_in_simdgroup]],
                     ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    gdn_rows_run(prep, lane_buf, state, g, args, faults, nullptr, nullptr, tgp, tpt, tiisg,
                 sgitg);
}

// gdn_rows with the copies `snaps` names: copy k's lane is `lane_len` floats
// of `snap` from k·lane_len, laid out as one lane of the state.
kernel void gdn_rows_snap(device const float *prep [[buffer(0)]],
                          device const uint *lane_buf [[buffer(1)]],
                          device float *state [[buffer(2)]],
                          device float *g [[buffer(3)]],
                          constant DeltaRowsArgs &args [[buffer(4)]],
                          device atomic_uint *faults [[buffer(5)]],
                          device float *snap [[buffer(6)]],
                          constant DeltaSnapArgs &snaps [[buffer(7)]],
                          uint3 tgp [[threadgroup_position_in_grid]],
                          uint3 tpt [[thread_position_in_threadgroup]],
                          ushort tiisg [[thread_index_in_simdgroup]],
                          ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    gdn_rows_run(prep, lane_buf, state, g, args, faults, snap, &snaps, tgp, tpt, tiisg, sgitg);
}

// The numerators and the partial sums of a chunk of rows, from the column
// outputs gdn_rows left in the numerator window. Grid: (value heads, rows)
// threadgroups of 128 threads, thread t the column t of the head. The
// numerator replaces the output in place.
kernel void gdn_post_rows(device const float *prep [[buffer(0)]],
                          device const float *norm_w [[buffer(1)]],
                          device float *partials [[buffer(2)]],
                          device float *g [[buffer(3)]],
                          constant DeltaRowsArgs &args [[buffer(4)]],
                          uint3 tgp [[threadgroup_position_in_grid]],
                          uint3 tpt [[thread_position_in_threadgroup]]) {
    uint head = BLOOMERY_DELTA_HEAD_COLS;
    uint h = tgp.x;
    uint r = tgp.y;
    uint tid = tpt.x;
    threadgroup float o2[128];
    device float *numerator = g + (ulong)r * args.g_stride + h * head;
    float oval = numerator[tid];
    o2[tid] = oval * oval;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < BLOOMERY_DELTA_COLUMN_BLOCKS) {
        partials[(ulong)r * args.partials_stride + h * BLOOMERY_DELTA_COLUMN_BLOCKS + tid] =
            gdn_block_sum(o2 + tid * BLOOMERY_DELTA_COLS_PER_BLOCK);
    }
    float gate = prep[(ulong)r * args.prep.stride + args.prep.gate + h * head + tid];
    numerator[tid] = gdn_numerator(oval, norm_w[tid], gate);
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

// The ring a chunk ending after row `after[k]` leaves, for each copy k: slot s
// holds the conv input of the latest position before the copy's own (the
// chunk's first position plus the row plus one) whose slot is s — a row of the
// chunk when that position is one, else the slot as the chunk found it. It
// reads the ring before gdn_ring_rows writes it, so the launch encodes it
// before that commit's barrier. Grid: the channels across, one row of the grid
// for each slot of each copy. Copy k's ring is `ring_len` floats of `snap`
// from k·ring_len, laid out as the ring.
kernel void gdn_ring_snap(device const float *x [[buffer(0)]],
                          device const float *ring [[buffer(1)]],
                          device float *snap [[buffer(2)]],
                          constant DeltaRingArgs &args [[buffer(3)]],
                          constant DeltaSnapArgs &snaps [[buffer(4)]],
                          uint2 gid [[thread_position_in_grid]]) {
    uint c = gid.x;
    uint s = gid.y % (uint)BLOOMERY_DELTA_RING_ROWS;
    uint k = gid.y / (uint)BLOOMERY_DELTA_RING_ROWS;
    if (c >= args.ch || k >= snaps.n) {
        return;
    }
    uint last = args.pos + snaps.after[k];
    // The slot's distance back from `last`, reduced first so no position wraps.
    uint back = (last % (uint)BLOOMERY_DELTA_RING_ROWS + (uint)BLOOMERY_DELTA_RING_ROWS - s)
                % (uint)BLOOMERY_DELTA_RING_ROWS;
    float v = ring[s * args.ch + c];
    if (back <= last && last - back >= args.pos) {
        v = x[(last - back - args.pos) * args.x_stride + c];
    }
    snap[(ulong)k * snaps.ring_len + s * args.ch + c] = v;
}
