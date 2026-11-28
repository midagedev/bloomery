// The elementwise tails of an Inkling 2 decode visit, run in the launch that
// makes their input. Each entry is a group of the walk that used to be two or
// three, and each channel's value is bit for bit what those launches wrote:
//
//   conv_res_join      the attention's post conv (conv_res) and the stream's
//                      add of it (bias_rows)
//   combine_conv_join  the experts' slots summed (combine_rows), the
//                      feed-forward's post conv of the sum (conv_res) and the
//                      stream's add of it (bias_rows)
//
// The conv is conv_res's rule at one token (conv_res_four), the ring slots and
// the filter loaded before the conv's input is used so their latency runs under
// the input's own, the ring stored after the reads. Every value of a channel
// depends on that channel alone, so a thread a float4 changes no bit.
//
// The slots' sum is combine_rows' own loop, a float4 a thread, the slots
// added in slot order from zero and the sum added to the visit's row once;
// every other value of a channel depends on that channel alone, so spreading
// the row over threads changes no bit. The sum comes after the conv's loads.
//
// The slots' rows are in slot order (the decode's route is the identity
// map), so slot s is row s of `y`, `y_stride` floats apart.
//
// No silent failure: a conv output that is not finite raises the linear-conv
// site and is NaN in the conv's row and in the stream (as conv_res's NaN is
// in the stream bias_rows adds it to).
//
// This file comes after fault.metal (raise), shortconv.metal (sc_slot) and
// conv_res.metal (conv_res_past4, conv_res_four).

#pragma METAL fp contract(off)

#define TAIL_SLOTS BLOOMERY_INK2_TOP_K

static_assert(TAIL_SLOTS == 8u, "the slots' sum is unrolled over the eight picks");

// The launch's argument block, bound by value.
struct CombineConvArgs {
    uint cols;       // the row's width, a multiple of 4
    uint y_stride;   // floats between the slots' rows
    uint ring_rows;  // the conv ring's rows
    uint ch;         // the ring's row width
    uint base;       // this conv's first channel in the ring's rows
    uint pos;        // this token's position
    uint layer;
    float scale;     // the conv's output scale
};

static_assert(sizeof(CombineConvArgs) == 32, "CombineConvArgs is 32 bytes: the host packs eight words");

// The conv-and-join launch's argument block, bound by value.
struct ConvJoinArgs {
    uint d;          // the conv's channels, a multiple of 4
    uint ring_rows;  // the conv ring's rows
    uint ch;         // the ring's row width
    uint base;       // this conv's first channel in the ring's rows
    uint pos;        // this token's position
    uint layer;
    float scale;     // the conv's output scale
};

static_assert(sizeof(ConvJoinArgs) == 28, "ConvJoinArgs is 28 bytes: the host packs seven words");

// y = the conv of x (into the ring too), stream += y. A thread a float4.
kernel void conv_res_join(device const float *x [[buffer(0)]],
                          device const float *w [[buffer(1)]],
                          device float *y [[buffer(2)]],
                          device float *ring [[buffer(3)]],
                          device float *stream [[buffer(4)]],
                          constant ConvJoinArgs &args [[buffer(5)]],
                          device atomic_uint *faults [[buffer(6)]],
                          uint gid [[thread_position_in_grid]])
{
    const uint f4 = gid;
    if (f4 >= args.d / 4u) {
        return;
    }
    const uint c = f4 * 4u;
    const uint at = args.base + c;

    const float4 cur = *((device const float4 *)x + f4);
    const device float4 *wv = (device const float4 *)w + c;
    const float4 t0 = wv[0];
    const float4 t1 = wv[1];
    const float4 t2 = wv[2];
    const float4 t3 = wv[3];
    const float4 p3 = conv_res_past4(ring, args.ring_rows, args.ch, at, args.pos, 3u);
    const float4 p2 = conv_res_past4(ring, args.ring_rows, args.ch, at, args.pos, 2u);
    const float4 p1 = conv_res_past4(ring, args.ring_rows, args.ch, at, args.pos, 1u);
    const float4 sr = *((device const float4 *)stream + f4);

    const float4 o = float4(conv_res_four(t0, p3.x, p2.x, p1.x, cur.x, args.scale),
                            conv_res_four(t1, p3.y, p2.y, p1.y, cur.y, args.scale),
                            conv_res_four(t2, p3.z, p2.z, p1.z, cur.z, args.scale),
                            conv_res_four(t3, p3.w, p2.w, p1.w, cur.w, args.scale));
    const bool ok = all(isfinite(o));
    const float4 out = select(float4(NAN), o, isfinite(o));
    *((device float4 *)y + f4) = out;
    *((device float4 *)(ring + (ulong)sc_slot(args.pos, args.ring_rows) * args.ch + at)) = cur;
    *((device float4 *)stream + f4) = sr + out;
    if (!ok) {
        raise(faults, args.layer, BLOOMERY_FAULT_SITE_LINEAR_CONV);
    }
}

// x[f4] += y[slot 0] + ... + y[slot 7] (slot order, from zero), the conv of the
// new x into conv_out and the ring, stream[f4] += the conv's output. A thread
// a float4.
kernel void combine_conv_join(device const float *y [[buffer(0)]],
                              device float *x [[buffer(1)]],
                              device const float *w [[buffer(2)]],
                              device float *conv_out [[buffer(3)]],
                              device float *ring [[buffer(4)]],
                              device float *stream [[buffer(5)]],
                              constant CombineConvArgs &args [[buffer(6)]],
                              device atomic_uint *faults [[buffer(7)]],
                              uint gid [[thread_position_in_grid]])
{
    const uint f4 = gid;
    if (f4 >= args.cols / 4u) {
        return;
    }
    const uint c = f4 * 4u;
    const uint at = args.base + c;

    // Every load of the thread, none depending on another: the slots' rows,
    // the visit's row, the filter's four channels, the ring's three
    // predecessors and the stream.
    float4 ys[TAIL_SLOTS];
    for (uint s = 0u; s < TAIL_SLOTS; ++s) {
        ys[s] = *((device const float4 *)(y + (ulong)s * args.y_stride) + f4);
    }
    const float4 xr = *((device const float4 *)x + f4);
    const device float4 *wv = (device const float4 *)w + c;
    const float4 t0 = wv[0];
    const float4 t1 = wv[1];
    const float4 t2 = wv[2];
    const float4 t3 = wv[3];
    const float4 p3 = conv_res_past4(ring, args.ring_rows, args.ch, at, args.pos, 3u);
    const float4 p2 = conv_res_past4(ring, args.ring_rows, args.ch, at, args.pos, 2u);
    const float4 p1 = conv_res_past4(ring, args.ring_rows, args.ch, at, args.pos, 1u);
    const float4 sr = *((device const float4 *)stream + f4);

    float4 acc = float4(0.f);
    for (uint s = 0u; s < TAIL_SLOTS; ++s) {
        acc += ys[s];
    }
    const float4 cur = xr + acc;
    *((device float4 *)x + f4) = cur;

    const float4 o = float4(conv_res_four(t0, p3.x, p2.x, p1.x, cur.x, args.scale),
                            conv_res_four(t1, p3.y, p2.y, p1.y, cur.y, args.scale),
                            conv_res_four(t2, p3.z, p2.z, p1.z, cur.z, args.scale),
                            conv_res_four(t3, p3.w, p2.w, p1.w, cur.w, args.scale));
    const bool ok = all(isfinite(o));
    const float4 out = select(float4(NAN), o, isfinite(o));
    *((device float4 *)conv_out + f4) = out;
    *((device float4 *)(ring + (ulong)sc_slot(args.pos, args.ring_rows) * args.ch + at)) = cur;
    *((device float4 *)stream + f4) = sr + out;
    if (!ok) {
        raise(faults, args.layer, BLOOMERY_FAULT_SITE_LINEAR_CONV);
    }
}
