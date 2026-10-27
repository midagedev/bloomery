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
// The step's grid is 256 threadgroups of 128 threads: 32 value heads, 8
// column blocks, 4 simdgroups. A simdgroup holds 4 columns of 8 lanes. Lane j
// of a column holds keys 32·r + 4·j + c. The q/k conv is redone in every
// threadgroup (256 channels, 4 taps).
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
// finite. gdn_rows is the step's 256 threadgroups with the state in
// registers across the rows: it reads the lane the word at `lane_at` names
// before the first row and writes the lane the word at `lane_to` names after
// the last, and a row reads only its prepared row. It writes the column
// outputs o into the numerator window, NaN where the row is unusable.
// gdn_post_rows turns them into the numerators and the partial sums of o². The
// ring is committed by gdn_ring_rows: the last min(m, ring) positions land in
// their slots, after the prepared rows were read.
//
// The recurrence of a rows launch runs in chunks of BLOOMERY_DELTA_CHUNK rows
// on simdgroup matrices (gdn_chunk_prep, gdn_chunk_rows below: the chunk form
// of the delta rule), which reorders the sums of the stepped launches: a row is
// within the f32 rounding of the stepped launch's, not its bits. What the
// chunked launch does not take (a lane word out of range, a chunk with a row
// whose status word is set, an output or state that is not finite) it leaves,
// from the first such chunk of a column block on, to gdn_rows, the row-by-row
// recurrence of the stepped launches, which reads the row the chunked launch
// stopped at from the first row's resume words and runs the rest in their bits.
//
// A rows launch may also leave copies after rows inside the chunk (a prompt
// pass's marks): the chunked launch ends a chunk after each named row and
// writes the state it holds into that copy's lane (gdn_rows_snap does the same
// row by row), and gdn_ring_snap writes the ring a chunk ending after that row
// would leave into that copy's ring, read before gdn_ring_rows overwrites the
// slots. A copy is the state a chunk ending at its row leaves; a cut moves the
// chunks after it, so the launch's own outputs move by rounding.
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
    uint aux;   // the chunk factors, GDN_AUX floats a value head
    uint resume; // one word a recurrence threadgroup, in the first row: the row the sequential launch starts at
};

static_assert(sizeof(GdnPrepLayout) == 36, "GdnPrepLayout is 36 bytes: the host packs nine words");

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

static_assert(sizeof(DeltaRowsArgs) == 96, "DeltaRowsArgs is 96 bytes: the host packs twenty-four words");

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

// The chunked recurrence. A chunk is up to GDN_C rows. With S the state
// (column × key), g_i the decay and β_i the weight of row i, k_i, q_i, v_i its
// key, query and value, Γ_i the product of the decays of rows 0..i of the
// chunk and M_ij the product of the decays of rows j+1..i (M_ii = 1), the
// rows of the chunk are
//
//     u = T R,  R_i = β_i (v_i − Γ_i S k_i),  T = (I + A)⁻¹,
//     A_ij = β_i M_ij (k_i · k_j) for j < i, else 0,
//     o_i = Γ_i S q_i + Σ_{j ≤ i} M_ij (q_i · k_j) u_j,
//     S' = Γ_{last} S + Σ_j M_{last,j} u_j k_jᵀ,
//
// the decay-first order of gdn_delta. The factors that do not read S (T, the
// score rows P_ij = M_ij (q_i · k_j), Γ and M_{last,j}) are computed for every
// chunk and value head in parallel by gdn_chunk_prep into the prepared rows'
// aux part; the products with S run in gdn_chunk_rows, which carries S in
// threadgroup memory across the chunks. A chunk ends at the row after a copy
// the launch leaves, so a copy is the state a chunk ends in.
//
// The chunked launch computes only what is finite and usable. A chunk whose
// rows carry a status word, whose output or new state is not finite, or a launch
// whose lane words are out of range, is not computed: the threadgroup writes
// the state it holds to the write lane (when it passed a chunk), records the
// row it stopped at, and gdn_rows takes every later row.

#define GDN_C ((uint)BLOOMERY_DELTA_CHUNK)
// The floats a row keeps for each value head: the row of T, the row of P,
// Γ_i, M_{last,i} and two pads.
#define GDN_AUX (2u * GDN_C + 4u)
// Row stride, in floats, of the threadgroup tiles of the chunk's factors.
#define GDN_TS (GDN_C + 4u)
// Row stride of the state tile (columns of a block by keys), and of a block's
// chunk-by-column tiles.
#define GDN_SS (BLOOMERY_DELTA_HEAD_COLS + 4u)
#define GDN_RS (BLOOMERY_DELTA_COLS_PER_BLOCK + 4u)

static_assert(BLOOMERY_DELTA_CHUNK % 8 == 0 && BLOOMERY_DELTA_CHUNK == 32,
              "the chunk is four tile rows: the tile loops below are written for 32 rows");
static_assert(BLOOMERY_DELTA_COLS_PER_BLOCK == 16, "a column block is two tile columns");
static_assert(BLOOMERY_DELTA_HEAD_COLS == 128, "a head is sixteen tile columns of keys");

// The chunk `index` of a launch of `rows` rows: its first row and length, or
// false past the last chunk. Chunks run from each cut (the row after a copy,
// and the launch's start) to the next, GDN_C rows at a time.
inline bool gdn_chunk_at(uint index, uint rows, constant DeltaSnapArgs &snaps,
                         thread uint *first, thread uint *len) {
    uint start = 0u;
    for (uint k = 0u; k <= snaps.n; ++k) {
        uint end = k < snaps.n ? snaps.after[k] + 1u : rows;
        uint chunks = (end - start + GDN_C - 1u) / GDN_C;
        if (index < chunks) {
            *first = start + index * GDN_C;
            *len = min(GDN_C, end - *first);
            return true;
        }
        index -= chunks;
        start = end;
    }
    return false;
}

// The factors of one chunk of rows for the value heads of a key head. Grid:
// (key heads, chunks) threadgroups of 128 threads. Each reads the prepared
// rows' keys, queries, β and decays and writes the aux part of the chunk's
// rows: for row i of head h, T_i, P_i, Γ_i, M_{last,i}. The tile products
// K Kᵀ and Q Kᵀ are shared by the key head's value heads. A row past the
// chunk's length is neither read nor written.
kernel void gdn_chunk_prep(device float *prep [[buffer(0)]],
                           constant DeltaRowsArgs &args [[buffer(1)]],
                           constant DeltaSnapArgs &snaps [[buffer(2)]],
                           uint3 tgp [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]],
                           ushort tiisg [[thread_index_in_simdgroup]],
                           ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    uint first;
    uint len;
    if (!gdn_chunk_at(tgp.y, args.rows, snaps, &first, &len)) {
        return;
    }
    uint n_v = BLOOMERY_DELTA_VALUE_HEADS;
    uint head = BLOOMERY_DELTA_HEAD_COLS;
    uint n_k = args.step.n_k;
    uint kh = tgp.x;
    uint per = n_v / n_k;
    ulong stride = args.prep.stride;
    device float *base = prep + (ulong)first * stride;
    device const float *kbase = base + args.prep.k + kh * head;
    device const float *qbase = base + kh * head;

    // The arrays are float4 storage seen as floats: the aux rows move as float4s.
    threadgroup float4 gs4[GDN_C * GDN_TS / 4u];
    threadgroup float4 qs4[GDN_C * GDN_TS / 4u];
    threadgroup float4 ms4[GDN_C * GDN_TS / 4u];
    threadgroup float4 am4[GDN_C * GDN_TS / 4u];
    threadgroup float4 ps4[GDN_C * GDN_TS / 4u];
    threadgroup float4 ts4[GDN_C * GDN_TS / 4u];
    threadgroup float *gs = (threadgroup float *)gs4;
    threadgroup float *qs = (threadgroup float *)qs4;
    threadgroup float *ms = (threadgroup float *)ms4;
    threadgroup float *amat = (threadgroup float *)am4;
    threadgroup float *ps = (threadgroup float *)ps4;
    threadgroup float *ts = (threadgroup float *)ts4;
    threadgroup float gv[GDN_C];
    threadgroup float bv[GDN_C];

    // K Kᵀ and Q Kᵀ, the lower tile triangle. Tile task t is (row tile, column
    // tile); a task whose tiles start past the chunk's length is skipped.
    const uint ti_of[10] = {0u, 1u, 1u, 2u, 2u, 2u, 3u, 3u, 3u, 3u};
    const uint tj_of[10] = {0u, 0u, 1u, 0u, 1u, 2u, 0u, 1u, 2u, 3u};
    for (uint t = sgitg; t < 10u; t += 4u) {
        uint ti = ti_of[t];
        uint tj = tj_of[t];
        if (8u * ti >= len || 8u * tj >= len) {
            continue;
        }
        simdgroup_float8x8 kk = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_float8x8 qk = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (uint kt = 0u; kt < head / 8u; ++kt) {
            simdgroup_float8x8 ak;
            simdgroup_float8x8 aq;
            simdgroup_float8x8 bk;
            simdgroup_load(ak, kbase + (8u * ti) * stride + 8u * kt, stride);
            simdgroup_load(aq, qbase + (8u * ti) * stride + 8u * kt, stride);
            simdgroup_load(bk, kbase + (8u * tj) * stride + 8u * kt, stride, ulong2(0), true);
            simdgroup_multiply_accumulate(kk, ak, bk, kk);
            simdgroup_multiply_accumulate(qk, aq, bk, qk);
        }
        simdgroup_store(kk, gs + 8u * ti * GDN_TS + 8u * tj, GDN_TS);
        simdgroup_store(qk, qs + 8u * ti * GDN_TS + 8u * tj, GDN_TS);
    }

    for (uint hi = 0u; hi < per; ++hi) {
        uint h = args.step.grouped != 0u ? kh * per + hi : kh + hi * n_k;
        // The tile stores above, and the previous head's reads of these
        // arrays, end here.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < GDN_C) {
            bool used = tid < len;
            gv[tid] = used ? (base + (ulong)tid * stride)[args.prep.decay + h] : 1.0f;
            bv[tid] = used ? (base + (ulong)tid * stride)[args.prep.beta + h] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // M, a column a lane: the product of the decays below the diagonal.
        if (tid < GDN_C) {
            float m = 1.0f;
            for (uint i = 0u; i < GDN_C; ++i) {
                if (i < tid) {
                    ms[i * GDN_TS + tid] = 0.0f;
                } else {
                    m = i == tid ? 1.0f : m * gv[i];
                    ms[i * GDN_TS + tid] = m;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = tid; e < GDN_C * GDN_C; e += 128u) {
            uint i = e / GDN_C;
            uint j = e % GDN_C;
            bool inside = i < len && j < len;
            amat[i * GDN_TS + j] = (inside && j < i) ? bv[i] * ms[i * GDN_TS + j] * gs[i * GDN_TS + j]
                                                  : 0.0f;
            ps[i * GDN_TS + j] = (inside && j <= i) ? ms[i * GDN_TS + j] * qs[i * GDN_TS + j] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // T = (I + A)⁻¹ by forward substitution, a column a lane.
        if (tid < GDN_C) {
            for (uint i = 0u; i < len; ++i) {
                float s0 = 0.0f;
                float s1 = 0.0f;
                float s2 = 0.0f;
                float s3 = 0.0f;
                uint l = 0u;
                for (; l + 4u <= i; l += 4u) {
                    s0 = fma(amat[i * GDN_TS + l], ts[l * GDN_TS + tid], s0);
                    s1 = fma(amat[i * GDN_TS + l + 1u], ts[(l + 1u) * GDN_TS + tid], s1);
                    s2 = fma(amat[i * GDN_TS + l + 2u], ts[(l + 2u) * GDN_TS + tid], s2);
                    s3 = fma(amat[i * GDN_TS + l + 3u], ts[(l + 3u) * GDN_TS + tid], s3);
                }
                for (; l < i; ++l) {
                    s0 = fma(amat[i * GDN_TS + l], ts[l * GDN_TS + tid], s0);
                }
                ts[i * GDN_TS + tid] = (i == tid ? 1.0f : 0.0f) - ((s0 + s1) + (s2 + s3));
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // The aux part of each row: T_i, P_i, then Γ_i and M_{last,i}.
        for (uint e = tid; e < len * (GDN_AUX / 4u); e += 128u) {
            uint i = e / (GDN_AUX / 4u);
            uint c = e % (GDN_AUX / 4u);
            float4 v;
            if (c < GDN_C / 4u) {
                v = *((threadgroup const float4 *)(ts + i * GDN_TS + 4u * c));
            } else if (c < GDN_C / 2u) {
                v = *((threadgroup const float4 *)(ps + i * GDN_TS + 4u * (c - GDN_C / 4u)));
            } else {
                v = float4(gv[0] * ms[i * GDN_TS], ms[(len - 1u) * GDN_TS + i], 0.0f, 0.0f);
            }
            *((device float4 *)(base + (ulong)i * stride + args.prep.aux + h * GDN_AUX + 4u * c)) = v;
        }
    }
}

// The state of a column block across the chunks of a launch, in threadgroup
// memory, and the chunk's products with it on simdgroup matrices: rows of the
// chunk are tile rows (simdgroup s the rows 8s..8s+7), the block's sixteen
// columns two tile columns, the keys sixteen tile columns (simdgroup s the
// keys 32s..32s+31 of the update). A prepared row is read through tiles, so
// the window holds the rows of the launch rounded up to a multiple of eight.
// Rows of a tile past the chunk's length are selected out, never multiplied
// into a result, but a product of the update may read them: a non-finite word
// there sends the chunk to the sequential launch.
inline void gdn_chunk_rows_run(device float *prep, device const uint *lane_buf, device float *state,
                               device float *g, constant DeltaRowsArgs &args, device float *snap,
                               constant DeltaSnapArgs *snaps, threadgroup float4 *s4,
                               threadgroup float4 *rs4, threadgroup float4 *vn4,
                               threadgroup float4 *vd4, threadgroup uint *sg_bad,
                               threadgroup uint *row_bad, uint3 tgp, uint tid, ushort tiisg,
                               ushort sgitg) {
    uint n_v = BLOOMERY_DELTA_VALUE_HEADS;
    uint head = BLOOMERY_DELTA_HEAD_COLS;
    uint tg = tgp.x;
    uint h = tg / BLOOMERY_DELTA_COLUMN_BLOCKS;
    uint block = tg % BLOOMERY_DELTA_COLUMN_BLOCKS;
    uint n_k = args.step.n_k;
    uint per = n_v / n_k;
    uint kh = args.step.grouped != 0u ? h / per : h % n_k;
    uint rows = args.rows;
    ulong stride = args.prep.stride;
    device uint *resume = (device uint *)(prep + args.prep.resume);

    uint word = lane_buf[args.step.lane_at];
    uint word_to = lane_buf[args.step.lane_to];
    if (word >= args.step.lanes || word_to >= args.step.lanes) {
        if (tid == 0u) {
            resume[tg] = 0u;
        }
        return;
    }
    uint base = ((word * n_v + h) * head + block * BLOOMERY_DELTA_COLS_PER_BLOCK) * head;
    uint base_to = ((word_to * n_v + h) * head + block * BLOOMERY_DELTA_COLS_PER_BLOCK) * head;
    uint cell = (h * head + block * BLOOMERY_DELTA_COLS_PER_BLOCK) * head;
    uint copies = snaps == nullptr ? 0u : snaps->n;
    uint copy = 0u;

    threadgroup float *sm = (threadgroup float *)s4;
    threadgroup float *rs = (threadgroup float *)rs4;
    threadgroup float *vn = (threadgroup float *)vn4;
    threadgroup float *vd = (threadgroup float *)vd4;

    // Where this lane sits in a tile: row `fm`, columns `fn` and `fn + 1`.
    ushort qid = ushort(tiisg / 4u);
    ushort fm = (qid & 4u) + ((tiisg / 2u) % 4u);
    ushort fn = (qid & 2u) * 2u + (tiisg % 2u) * 2u;

    for (uint e = tid; e < BLOOMERY_DELTA_COLS_PER_BLOCK * (BLOOMERY_DELTA_HEAD_COLS / 4u); e += 128u) {
        uint row = e / (BLOOMERY_DELTA_HEAD_COLS / 4u);
        uint c4 = e % (BLOOMERY_DELTA_HEAD_COLS / 4u);
        *((threadgroup float4 *)(sm + row * GDN_SS + 4u * c4)) =
            *((device const float4 *)(state + base + row * head + 4u * c4));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint r = 0u;
    while (r < rows) {
        uint cut = copy < copies ? snaps->after[copy] + 1u : rows;
        uint len = min(GDN_C, cut - r);
        device const float *rowp = prep + (ulong)r * stride;
        device const float *aux = rowp + args.prep.aux + h * GDN_AUX;
        bool live = 8u * sgitg < len;

        if (sgitg == 0u) {
            uint w = 0u;
            if (tiisg < len) {
                w = ((device const uint *)(rowp + (ulong)tiisg * stride + args.prep.bad))[tg];
            }
            w = simd_or(w);
            if (tiisg == 0u) {
                *row_bad = w;
            }
        }

        // K Sᵀ and Q Sᵀ for this simdgroup's tile row.
        simdgroup_float8x8 ky[2];
        simdgroup_float8x8 qy[2];
        ky[0] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        ky[1] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        qy[0] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        qy[1] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        if (live) {
            device const float *krow = rowp + (ulong)(8u * sgitg) * stride + args.prep.k + kh * head;
            device const float *qrow = rowp + (ulong)(8u * sgitg) * stride + kh * head;
            _Pragma("clang loop unroll(full)")
            for (uint kt = 0u; kt < head / 8u; ++kt) {
                simdgroup_float8x8 ak;
                simdgroup_float8x8 aq;
                simdgroup_float8x8 b0;
                simdgroup_float8x8 b1;
                simdgroup_load(ak, krow + 8u * kt, stride);
                simdgroup_load(aq, qrow + 8u * kt, stride);
                simdgroup_load(b0, sm + 8u * kt, GDN_SS, ulong2(0), true);
                simdgroup_load(b1, sm + 8u * GDN_SS + 8u * kt, GDN_SS, ulong2(0), true);
                simdgroup_multiply_accumulate(ky[0], ak, b0, ky[0]);
                simdgroup_multiply_accumulate(ky[1], ak, b1, ky[1]);
                simdgroup_multiply_accumulate(qy[0], aq, b0, qy[0]);
                simdgroup_multiply_accumulate(qy[1], aq, b1, qy[1]);
            }
        }

        // R = β (v − Γ K Sᵀ); the query term scaled by Γ.
        uint ri = 8u * sgitg + fm;
        bool inside = ri < len;
        device const float *mine = rowp + (ulong)ri * stride;
        float beta = inside ? mine[args.prep.beta + h] : 0.0f;
        float gamma = inside ? mine[args.prep.aux + h * GDN_AUX + 2u * GDN_C] : 0.0f;
        _Pragma("clang loop unroll(full)")
        for (uint ct = 0u; ct < 2u; ++ct) {
            float2 v = inside ? *((device const float2 *)(mine + args.prep.v + h * head +
                                                          block * BLOOMERY_DELTA_COLS_PER_BLOCK +
                                                          8u * ct + fn))
                              : float2(0.0f);
            simdgroup_float8x8 rt;
            rt.thread_elements()[0] = inside ? beta * (v.x - gamma * ky[ct].thread_elements()[0]) : 0.0f;
            rt.thread_elements()[1] = inside ? beta * (v.y - gamma * ky[ct].thread_elements()[1]) : 0.0f;
            simdgroup_store(rt, rs + 8u * sgitg * GDN_RS + 8u * ct, GDN_RS);
            qy[ct].thread_elements()[0] *= gamma;
            qy[ct].thread_elements()[1] *= gamma;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // u = T R, kept in two forms: the rows themselves for the scores, and
        // the rows weighted by M_{last,j} for the update.
        simdgroup_float8x8 un[2];
        un[0] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        un[1] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        if (live) {
            device const float *trow = aux + (ulong)(8u * sgitg) * stride;
            for (uint lb = 0u; lb <= sgitg; ++lb) {
                simdgroup_float8x8 tt;
                simdgroup_float8x8 r0;
                simdgroup_float8x8 r1;
                simdgroup_load(tt, trow + 8u * lb, stride);
                simdgroup_load(r0, rs + 8u * lb * GDN_RS, GDN_RS);
                simdgroup_load(r1, rs + 8u * lb * GDN_RS + 8u, GDN_RS);
                simdgroup_multiply_accumulate(un[0], tt, r0, un[0]);
                simdgroup_multiply_accumulate(un[1], tt, r1, un[1]);
            }
        }
        float mend = inside ? mine[args.prep.aux + h * GDN_AUX + 2u * GDN_C + 1u] : 0.0f;
        _Pragma("clang loop unroll(full)")
        for (uint ct = 0u; ct < 2u; ++ct) {
            float u0 = inside ? un[ct].thread_elements()[0] : 0.0f;
            float u1 = inside ? un[ct].thread_elements()[1] : 0.0f;
            simdgroup_float8x8 a;
            simdgroup_float8x8 b;
            a.thread_elements()[0] = u0;
            a.thread_elements()[1] = u1;
            b.thread_elements()[0] = mend * u0;
            b.thread_elements()[1] = mend * u1;
            simdgroup_store(a, vn + 8u * sgitg * GDN_RS + 8u * ct, GDN_RS);
            simdgroup_store(b, vd + 8u * sgitg * GDN_RS + 8u * ct, GDN_RS);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // o = Γ Q Sᵀ + P u, to the numerator window.
        bool unusable = false;
        if (live) {
            device const float *prow = aux + (ulong)(8u * sgitg) * stride + GDN_C;
            for (uint lb = 0u; lb <= sgitg; ++lb) {
                simdgroup_float8x8 pp;
                simdgroup_float8x8 u0;
                simdgroup_float8x8 u1;
                simdgroup_load(pp, prow + 8u * lb, stride);
                simdgroup_load(u0, vn + 8u * lb * GDN_RS, GDN_RS);
                simdgroup_load(u1, vn + 8u * lb * GDN_RS + 8u, GDN_RS);
                simdgroup_multiply_accumulate(qy[0], pp, u0, qy[0]);
                simdgroup_multiply_accumulate(qy[1], pp, u1, qy[1]);
            }
            _Pragma("clang loop unroll(full)")
            for (uint ct = 0u; ct < 2u; ++ct) {
                float o0 = qy[ct].thread_elements()[0];
                float o1 = qy[ct].thread_elements()[1];
                if (inside) {
                    *((device float2 *)(g + (ulong)(r + ri) * args.g_stride + h * head +
                                        block * BLOOMERY_DELTA_COLS_PER_BLOCK + 8u * ct + fn)) =
                        float2(o0, o1);
                    unusable = unusable || !isfinite(o0) || !isfinite(o1);
                }
            }
        }

        // S' = Γ_last S + (M_last u)ᵀ K, each simdgroup its 32 keys.
        float last = (prep + (ulong)(r + len - 1u) * stride)[args.prep.aux + h * GDN_AUX + 2u * GDN_C];
        simdgroup_float8x8 ns[2][4];
        uint jblocks = (len + 7u) / 8u;
        _Pragma("clang loop unroll(full)")
        for (uint ct = 0u; ct < 2u; ++ct) {
            _Pragma("clang loop unroll(full)")
            for (uint q = 0u; q < 4u; ++q) {
                uint kt = 4u * sgitg + q;
                simdgroup_float8x8 acc;
                simdgroup_load(acc, sm + 8u * ct * GDN_SS + 8u * kt, GDN_SS);
                acc.thread_elements()[0] *= last;
                acc.thread_elements()[1] *= last;
                for (uint jb = 0u; jb < jblocks; ++jb) {
                    simdgroup_float8x8 av;
                    simdgroup_float8x8 bk;
                    simdgroup_load(av, vd + 8u * jb * GDN_RS + 8u * ct, GDN_RS, ulong2(0), true);
                    simdgroup_load(bk, rowp + (ulong)(8u * jb) * stride + args.prep.k + kh * head + 8u * kt,
                                   stride);
                    simdgroup_multiply_accumulate(acc, av, bk, acc);
                }
                unusable = unusable || !isfinite(acc.thread_elements()[0]) ||
                           !isfinite(acc.thread_elements()[1]);
                ns[ct][q] = acc;
            }
        }
        unusable = simd_any(unusable);
        if (tiisg == 0u) {
            sg_bad[sgitg] = unusable ? 1u : 0u;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if ((*row_bad | sg_bad[0] | sg_bad[1] | sg_bad[2] | sg_bad[3]) != 0u) {
            if (r > 0u) {
                for (uint e = tid; e < BLOOMERY_DELTA_COLS_PER_BLOCK * (BLOOMERY_DELTA_HEAD_COLS / 4u);
                     e += 128u) {
                    uint row = e / (BLOOMERY_DELTA_HEAD_COLS / 4u);
                    uint c4 = e % (BLOOMERY_DELTA_HEAD_COLS / 4u);
                    *((device float4 *)(state + base_to + row * head + 4u * c4)) =
                        *((threadgroup float4 *)(sm + row * GDN_SS + 4u * c4));
                }
            }
            if (tid == 0u) {
                resume[tg] = r;
            }
            return;
        }
        _Pragma("clang loop unroll(full)")
        for (uint ct = 0u; ct < 2u; ++ct) {
            _Pragma("clang loop unroll(full)")
            for (uint q = 0u; q < 4u; ++q) {
                simdgroup_store(ns[ct][q], sm + 8u * ct * GDN_SS + 8u * (4u * sgitg + q), GDN_SS);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        r += len;
        if (copy < copies && r == cut) {
            for (uint e = tid; e < BLOOMERY_DELTA_COLS_PER_BLOCK * (BLOOMERY_DELTA_HEAD_COLS / 4u);
                 e += 128u) {
                uint row = e / (BLOOMERY_DELTA_HEAD_COLS / 4u);
                uint c4 = e % (BLOOMERY_DELTA_HEAD_COLS / 4u);
                *((device float4 *)(snap + (ulong)copy * snaps->lane_len + cell + row * head + 4u * c4)) =
                    *((threadgroup float4 *)(sm + row * GDN_SS + 4u * c4));
            }
            ++copy;
        }
    }
    for (uint e = tid; e < BLOOMERY_DELTA_COLS_PER_BLOCK * (BLOOMERY_DELTA_HEAD_COLS / 4u); e += 128u) {
        uint row = e / (BLOOMERY_DELTA_HEAD_COLS / 4u);
        uint c4 = e % (BLOOMERY_DELTA_HEAD_COLS / 4u);
        *((device float4 *)(state + base_to + row * head + 4u * c4)) =
            *((threadgroup float4 *)(sm + row * GDN_SS + 4u * c4));
    }
    if (tid == 0u) {
        resume[tg] = rows;
    }
}

// The threadgroup memory of the chunked recurrence: the state tile and the
// chunk's tiles as float4 storage (the state moves between the lane and the
// tile as float4s), and the status words.
#define GDN_CHUNK_SHARED                                                          \
    threadgroup float4 s4[BLOOMERY_DELTA_COLS_PER_BLOCK * GDN_SS / 4u];            \
    threadgroup float4 rs4[GDN_C * GDN_RS / 4u];                                   \
    threadgroup float4 vn4[GDN_C * GDN_RS / 4u];                                   \
    threadgroup float4 vd4[GDN_C * GDN_RS / 4u];                                   \
    threadgroup uint sg_bad[4];                                                    \
    threadgroup uint row_bad;

kernel void gdn_chunk_rows(device float *prep [[buffer(0)]],
                           device const uint *lane_buf [[buffer(1)]],
                           device float *state [[buffer(2)]],
                           device float *g [[buffer(3)]],
                           constant DeltaRowsArgs &args [[buffer(4)]],
                           uint3 tgp [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]],
                           ushort tiisg [[thread_index_in_simdgroup]],
                           ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    GDN_CHUNK_SHARED
    gdn_chunk_rows_run(prep, lane_buf, state, g, args, nullptr, nullptr, s4, rs4, vn4, vd4, sg_bad,
                       &row_bad, tgp, tid, tiisg, sgitg);
}

kernel void gdn_chunk_rows_snap(device float *prep [[buffer(0)]],
                                device const uint *lane_buf [[buffer(1)]],
                                device float *state [[buffer(2)]],
                                device float *g [[buffer(3)]],
                                constant DeltaRowsArgs &args [[buffer(4)]],
                                device float *snap [[buffer(5)]],
                                constant DeltaSnapArgs &snaps [[buffer(6)]],
                                uint3 tgp [[threadgroup_position_in_grid]],
                                uint tid [[thread_index_in_threadgroup]],
                                ushort tiisg [[thread_index_in_simdgroup]],
                                ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    GDN_CHUNK_SHARED
    gdn_chunk_rows_run(prep, lane_buf, state, g, args, snap, &snaps, s4, rs4, vn4, vd4, sg_bad,
                       &row_bad, tgp, tid, tiisg, sgitg);
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

    // The chunked launch leaves the row its threadgroup stopped at: this
    // launch takes the rows from there, from the lane that launch left the
    // state in (the read lane when it stopped before any row).
    uint start = ((device const uint *)(prep + args.prep.resume))[tgp.x];
    if (start >= args.rows) {
        return;
    }
    uint word = lane_buf[args.step.lane_at];
    uint word_to = lane_buf[args.step.lane_to];
    bool lane_ok = word < args.step.lanes && word_to < args.step.lanes;
    uint from = start == 0u ? word : word_to;
    uint base = ((from * n_v + seat.h) * head + seat.col) * head;
    uint base_to = ((word_to * n_v + seat.h) * head + seat.col) * head;
    uint cell = (seat.h * head + seat.col) * head;
    uint copies = snaps == nullptr ? 0u : snaps->n;
    uint copy = 0u;
    while (copy < copies && snaps->after[copy] < start) {
        ++copy;
    }

    float st[16];
    gdn_load(state, base, seat.j, lane_ok, st);
    bool row_ok = lane_ok;
    for (uint r = start; r < args.rows; ++r) {
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
