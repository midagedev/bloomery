// The short conv of a gated-conv mixer: for each channel c,
//
//   y_t[c] = C_t[c] · Σ_k w[c][k] · u_{t-(taps-1)+k}[c],   u_t = B_t ⊙ x_t
//
// causal and depthwise, oldest tap first, no bias and no activation. A row of
// the input is the in_proj output, B | C | x, `hidden` values each; the weight
// is [hidden][taps], a channel's taps together. The tap count and the ring's
// rows are arguments, so any conv width runs.
//
// The ring stores u, not the projection: the last positions of one sequence,
// the slot of a position being (position mod ring_rows). The current token is
// the launch's own row, never a ring slot. A predecessor `back` positions
// before a token is, in this order:
//   - 0 while its position is before the sequence's first, whatever the ring
//     holds;
//   - the chunk's own row when the chunk holds it;
//   - the ring slot of its position otherwise.
// The ring holds at least `taps` rows (the host refuses a smaller ring), so the
// slot a token writes is not a slot it reads.
//
// Three launches, each one thread a (channel, row):
//   - shortconv_step: one row per slot. Row s is slot s's token, at the
//     position the slot's word of `pos` names, over slot s's ring (slot s's
//     rings are contiguous: ring_rows·hidden floats each). The thread reads
//     its predecessors, then writes its token's u into its ring slot.
//   - shortconv_rows: the rows of one sequence's chunk, from position `pos`.
//     A predecessor inside the chunk is that row's u, one before the chunk
//     is the ring's. It reads the ring and writes none.
//   - shortconv_ring_rows: the ring after a chunk. The last min(rows,
//     ring_rows) positions' u land in their slots, after the rows launch
//     read the pre-chunk slots; the caller orders the two with a barrier.
// A chunk's rows are the bits of stepped launches at their positions: both
// launches call the same inline functions.
//
// A value that is not finite (an input, a weight, the sum or the output)
// raises the linear-conv site and the output is NaN, never a plausible value.
// The ring commit stores u as it is, so a non-finite u is the ring's too:
// the fault word stays raised until the model resets.
//
// This file comes after fault.metal, whose `raise` and site table it reads.

#include <metal_stdlib>

using namespace metal;

#pragma METAL fp contract(off)

// The launch's argument block, bound by value. `rows` is the slots of a step
// and the rows of a chunk; `pos` is the chunk's first position (a step takes
// its positions from the buffer and ignores the word).
struct ShortConvArgs {
    uint taps;
    uint ring_rows;
    uint hidden;
    uint rows;
    uint x_stride; // floats between rows of the input
    uint y_stride; // floats between rows of the output
    uint layer;
    uint pos;
};

static_assert(sizeof(ShortConvArgs) == 32, "ShortConvArgs is 32 bytes: the host packs eight words");

// The ring slot of a position.
inline uint sc_slot(uint pos, uint ring_rows) {
    return pos % ring_rows;
}

// u = B ⊙ x of row r of the input, channel c: the f32 product the ring keeps.
inline float sc_u(device const float *bcx, uint r, uint c, constant ShortConvArgs &a) {
    device const float *row = bcx + r * a.x_stride;
    return row[c] * row[2u * a.hidden + c];
}

// The u `back` positions before row r of a chunk that starts at position
// pos0. back 0 is this row's own.
inline float sc_tap(device const float *bcx, device const float *ring, uint c, uint pos0, uint r,
                    uint back, constant ShortConvArgs &a) {
    if (back <= r) {
        return sc_u(bcx, r - back, c, a);
    }
    uint before = back - r;
    if (pos0 < before) {
        return 0.0f;
    }
    return ring[sc_slot(pos0 - before, a.ring_rows) * a.hidden + c];
}

// The conv of row r, channel c, and whether every value stayed finite: the
// taps' products in one fma chain from the oldest, then the C gate.
inline float sc_conv(device const float *bcx, device const float *ring, device const float *w,
                     uint c, uint pos0, uint r, constant ShortConvArgs &a, thread bool *ok) {
    float acc = 0.0f;
    for (uint k = 0u; k < a.taps; ++k) {
        float u = sc_tap(bcx, ring, c, pos0, r, a.taps - 1u - k, a);
        float wk = w[c * a.taps + k];
        acc = fma(u, wk, acc);
        if (!isfinite(u) || !isfinite(wk)) {
            *ok = false;
        }
    }
    float y = acc * bcx[r * a.x_stride + a.hidden + c];
    if (!isfinite(acc) || !isfinite(y)) {
        *ok = false;
    }
    return y;
}

// The output word: the value, or NaN for a row element that was not usable.
inline float sc_out(bool ok, float v) {
    return ok ? v : NAN;
}

inline void sc_raise(device atomic_uint *faults, uint layer) {
    raise(faults, layer, BLOOMERY_FAULT_SITE_LINEAR_CONV);
}

kernel void shortconv_step(device const float *bcx [[buffer(0)]],
                           device const float *w [[buffer(1)]],
                           device float *ring [[buffer(2)]],
                           device float *y [[buffer(3)]],
                           constant ShortConvArgs &args [[buffer(4)]],
                           constant uint *pos [[buffer(5)]],
                           device atomic_uint *faults [[buffer(6)]],
                           uint2 gid [[thread_position_in_grid]]) {
    const uint c = gid.x;
    const uint s = gid.y;
    if (c >= args.hidden || s >= args.rows) {
        return;
    }
    device const float *row = bcx + s * args.x_stride;
    device float *slot_ring = ring + s * args.ring_rows * args.hidden;
    const uint p = pos[s];
    bool ok = true;
    float v = sc_conv(row, slot_ring, w, c, p, 0u, args, &ok);
    y[s * args.y_stride + c] = sc_out(ok, v);
    if (!ok) {
        sc_raise(faults, args.layer);
    }
    slot_ring[sc_slot(p, args.ring_rows) * args.hidden + c] = sc_u(row, 0u, c, args);
}

kernel void shortconv_rows(device const float *bcx [[buffer(0)]],
                           device const float *w [[buffer(1)]],
                           device const float *ring [[buffer(2)]],
                           device float *y [[buffer(3)]],
                           constant ShortConvArgs &args [[buffer(4)]],
                           device atomic_uint *faults [[buffer(5)]],
                           uint2 gid [[thread_position_in_grid]]) {
    const uint c = gid.x;
    const uint r = gid.y;
    if (c >= args.hidden || r >= args.rows) {
        return;
    }
    bool ok = true;
    float v = sc_conv(bcx, ring, w, c, args.pos, r, args, &ok);
    y[r * args.y_stride + c] = sc_out(ok, v);
    if (!ok) {
        sc_raise(faults, args.layer);
    }
}

// The ring after a chunk: thread (c, j) writes the u of the chunk's row
// rows - kept + j, kept = min(rows, ring_rows). Those are consecutive
// positions, at most ring_rows of them, so no two share a slot.
kernel void shortconv_ring_rows(device const float *bcx [[buffer(0)]],
                                device float *ring [[buffer(1)]],
                                constant ShortConvArgs &args [[buffer(2)]],
                                uint2 gid [[thread_position_in_grid]]) {
    const uint c = gid.x;
    const uint kept = min(args.rows, args.ring_rows);
    if (c >= args.hidden || gid.y >= kept) {
        return;
    }
    const uint r = args.rows - kept + gid.y;
    ring[sc_slot(args.pos + r, args.ring_rows) * args.hidden + c] = sc_u(bcx, r, c, args);
}
