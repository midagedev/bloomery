// The residual short conv of an Inkling 2 visit: a causal depthwise conv
// over the last `taps` inputs of each channel, the current input added as a
// residual, the whole scaled by 1/sqrt((dynamic ? 2 : 1)·taps + 1), and on
// the windowless layers' k and v convs a second set of input-dependent taps
// beside the static ones. The projections around it are the tree's dense
// mat-vecs and are not here.
//
// The rule, per token t of the call and channel c of `d` (the reference's
// op order, each tap loop its own so the bits match term for term):
//
//   acc  = w[0, c] · x(t, taps−1)                       the oldest static tap first
//   acc  = fma(w[k, c], x(t, taps−1−k), acc)            k = 1 .. taps−1
//   acc  = fma(dyn[t, h(c), lag], x(t, lag), acc)       the dynamic taps, lag 0 the current
//   y    = (x(t, 0) + acc) · scale                      the residual cur-first, the scale last
//
// The static and dynamic taps index opposite ends of the window exactly as
// the reference spells them: static w[0] pairs the oldest input, dynamic
// dyn[·, ·, 0] the current one. The two sums are separate accumulations in
// that order, the residual joins after both, the scale last.
//
// Layouts, all f32 and token-major:
//   - the input x [m][d] — the projection's raw output rows, what the ring
//     stores;
//   - the static filters w [d][taps] as the file lays it (channel c's taps
//     contiguous at c · taps, the oldest first);
//   - the dynamic taps dyn [m][heads][taps] (token-major, head, then the
//     taps contiguous, lag 0 the current input); `heads` 0 marks a static
//     conv and the buffer may be empty;
//   - y [m][d];
//   - the conv ring [rows][ch] over the visit's whole channel width: the
//     four convs of a visit read and write disjoint channel ranges of the
//     one ring (the reference's single recurrent buffer), so the launch
//     takes the ring's row width `ch` and this conv's channel base in it
//     (base + d <= ch), and keeps the ring's indexing: the input of
//     position p in slot p mod rows.
//
// Positions: token t sits at position pos + t, the call's first position
// the launch's own word. A token's predecessor `back` positions before (1
// to taps − 1) is the call's row t − back when back <= t; otherwise it lies
// before the call: zero when pos + t < back (the sequence start, whatever
// the ring holds), else the ring slot (pos + t − back) mod rows. The call
// writes the input of each of its last min(m, rows) tokens into slot
// (pos + t) mod rows, after every read of the channel's ring.
//
// Geometry: one thread a (channel, unit), the channel the fast index, 128
// threads a threadgroup, no threadgroup memory and no barrier — unit 0 the
// first min(m, taps − 1) tokens (the ones that read the ring), in order,
// and afterwards the ring store; unit u >= 1 the one token taps − 2 + u,
// whose predecessors are all the call's rows. The ring is touched by unit 0
// alone, each channel by one thread, its reads before its writes.
//
// No silent failure: an output that is not finite (an input or a tap
// product was not) raises the linear-conv site and the value written there
// is NaN. The ring is stored as the inputs are: an input that is not
// finite stays in the ring and every later call that reads it raises again,
// until the model resets.
//
// This file comes after fault.metal (raise) and shortconv.metal (sc_slot,
// the ring slot rule the two convs share).

#pragma METAL fp contract(off)

static_assert(128u <= 1024u, "the conv's threadgroup is one thread a channel");

// The launch's argument block, bound by value.
struct ConvResArgs {
    uint d;      // the conv's channels
    uint taps;   // the filter's taps, the current position's among them
    uint heads;  // the dynamic taps' heads, 0 a static conv
    uint rows;   // the ring's rows
    uint ch;     // the ring's row width (the visit's whole channel width)
    uint base;   // the conv's channel base in the ring's rows
    uint m;      // tokens
    uint layer;
    uint pos;    // the call's first position
    float scale; // 1/sqrt((heads ? 2 : 1) · taps + 1), the caller derives
};

static_assert(sizeof(ConvResArgs) == 40, "ConvResArgs is 40 bytes: the host packs ten words");

// The input `back` positions before token t of the call: the call's own
// row when it has one, else the ring's, zero at the sequence's start.
inline float conv_res_tap(device const float *x,
                          device const float *ring,
                          uint d,
                          uint ch,
                          uint base,
                          uint rows,
                          uint t,
                          uint pos0,
                          uint back,
                          uint c)
{
    if (back <= t) {
        return x[(t - back) * d + c];
    }
    // The predecessor lies before the call, `before` positions ahead of its
    // first: nothing there when the call starts inside the taps.
    const uint before = back - t;
    if (before > pos0) {
        return 0.0f;
    }
    return ring[sc_slot(pos0 - before, rows) * ch + base + c];
}

// The decode token's conv, for the entries that run it in the tail of the
// launch that makes its input (conv_tail.metal, qk_clip.metal): a four-tap
// static filter (and, on a windowless k or v conv, a head's four dynamic
// taps), token 0 of a call of one, so the oldest static tap pairs the input
// three positions back and the newest the current one. One owner of the rule
// beside `conv_res` for those entries; the clauses hold each of them to
// `conv_res`'s bits.

// The ring's input `back` positions before position `pos`, channel `at` of the
// ring's rows of `ch`: zero before the sequence's start, whatever the ring holds.
inline float conv_res_past(device const float *ring, uint rows, uint ch, uint at, uint pos, uint back)
{
    if (back > pos) {
        return 0.0f;
    }
    return ring[(ulong)sc_slot(pos - back, rows) * ch + at];
}

// The same for four channels from `at`, a multiple of 4 on a 16-byte boundary.
inline float4 conv_res_past4(device const float *ring, uint rows, uint ch, uint at, uint pos, uint back)
{
    if (back > pos) {
        return float4(0.0f);
    }
    return *((device const float4 *)(ring + (ulong)sc_slot(pos - back, rows) * ch + at));
}

// The static taps oldest first: the first product rounds alone, the rest fuse
// ascending. `tap` is the channel's four filter taps, `p3`, `p2`, `p1` its
// inputs three, two and one positions back and `cur` its input now.
inline float conv_res_acc4(float4 tap, float p3, float p2, float p1, float cur)
{
    float acc = tap.x * p3;
    acc = fma(tap.y, p2, acc);
    acc = fma(tap.z, p1, acc);
    return fma(tap.w, cur, acc);
}

// One channel's output of a static conv: the residual cur-first, the scale last.
inline float conv_res_four(float4 tap, float p3, float p2, float p1, float cur, float scale)
{
    return (cur + conv_res_acc4(tap, p3, p2, p1, cur)) * scale;
}

// One channel's output of a conv with dynamic taps: after the static taps,
// `dy` (its head's four taps, lag 0 the current input) fuse in lag order.
inline float conv_res_four_dyn(float4 tap, float4 dy, float p3, float p2, float p1, float cur, float scale)
{
    float acc = conv_res_acc4(tap, p3, p2, p1, cur);
    acc = fma(dy.x, cur, acc);
    acc = fma(dy.y, p1, acc);
    acc = fma(dy.z, p2, acc);
    acc = fma(dy.w, p3, acc);
    return (cur + acc) * scale;
}

kernel void conv_res(device const float *x [[buffer(0)]],
                     device const float *w [[buffer(1)]],
                     device const float *dyn [[buffer(2)]],
                     device float *y [[buffer(3)]],
                     device float *ring [[buffer(4)]],
                     constant ConvResArgs &args [[buffer(5)]],
                     device atomic_uint *faults [[buffer(6)]],
                     uint2 gid [[thread_position_in_grid]])
{
    const uint c = gid.x;
    const uint d = args.d;
    const uint taps = args.taps;
    const uint reach = taps - 1u;
    // Unit 0 holds the call's first reach tokens and the ring store; unit
    // u >= 1 holds the one token reach - 1 + u. A call shorter than the
    // reach is unit 0 alone.
    const uint units = args.m > reach ? 1u + args.m - reach : 1u;
    if (c >= d || gid.y >= units) {
        return;
    }
    const uint u = gid.y;
    const uint per_head = args.heads > 0u ? d / args.heads : 1u;
    bool ok = true;

    const uint t0 = u == 0u ? 0u : reach - 1u + u;
    const uint t1 = u == 0u ? min(args.m, reach) : t0 + 1u;
    for (uint t = t0; t < t1; ++t) {
        // The static taps, the oldest first: tap k reads the input
        // taps − 1 − k positions back.
        float acc = w[c * taps] *
                    conv_res_tap(x, ring, d, args.ch, args.base, args.rows, t, args.pos, reach, c);
        for (uint k = 1u; k < taps; ++k) {
            acc = fma(w[c * taps + k],
                      conv_res_tap(x, ring, d, args.ch, args.base, args.rows, t, args.pos, reach - k, c),
                      acc);
        }
        // The dynamic taps, the current input first: lag reads the input
        // lag positions back, its head's own row of the call.
        if (args.heads > 0u) {
            const uint head = c / per_head;
            for (uint lag = 0u; lag < taps; ++lag) {
                acc = fma(dyn[(t * args.heads + head) * taps + lag],
                          conv_res_tap(x, ring, d, args.ch, args.base, args.rows, t, args.pos, lag, c),
                          acc);
            }
        }
        const float out = (x[t * d + c] + acc) * args.scale;
        ok = ok && isfinite(out);
        y[t * d + c] = isfinite(out) ? out : NAN;
    }
    // Unit 0 stores the ring after its reads: the call's last
    // min(m, rows) inputs land in their slots.
    if (u == 0u) {
        const uint kept = min(args.m, args.rows);
        for (uint j = 0u; j < kept; ++j) {
            const uint t = args.m - kept + j;
            ring[sc_slot(args.pos + t, args.rows) * args.ch + args.base + c] = x[t * d + c];
        }
    }
    if (!ok) {
        raise(faults, args.layer, BLOOMERY_FAULT_SITE_LINEAR_CONV);
    }
}
