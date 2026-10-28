// What every mat-vec of the quantized kernels shares on the activation side:
// the argument block, the RMS norm a mat-vec applies as it loads its input,
// and the SIMD groups in a mat-vec's threadgroup.
//
// The norm is not a pass of its own. Every SIMD group of a mat-vec loads the
// whole input vector once (each lane its own disjoint values), so the same
// loads give the sum of squares: the lane adds the squares of what it loaded,
// the SIMD group sums them, and the scale `1 / sqrt(mean + eps)` multiplies
// the finished dot product. The gain is applied to each value as it is
// loaded, the scale once at the end: the dot product is linear in the input,
// so `W . (x * gain * scale) == scale * (W . (x * gain))`.
//
// This file comes before the mat-vec files in the concatenated source; it
// needs the fault site's table and `raise` (fault.metal).

#pragma METAL fp contract(off)

// The argument block of a mat-vec, bound by value at slot 3.
struct MatvecArgs {
    uint  rows;     // output rows
    uint  nblocks;  // input columns / the type's block values
    uint  layer;    // the layer a raise is recorded in (BLOOMERY_LAYER_NONE for a direct launch)
    float eps;      // the norm's epsilon (read by the norm variants only)
};

static_assert(sizeof(MatvecArgs) == 16, "MatvecArgs is 16 bytes: the host packs four words");

// One segment of a segmented mat-vec: the rows of one tensor, and where they
// sit in the weights and in the output.
struct MatvecSeg {
    uint rows;         // the segment's output rows
    uint wblocks;      // the segment's first row's block offset in the weights
    uint out;          // the segment's first row's element offset in the output
    uint first_group;  // the first threadgroup of the segment
};

// The argument block of a segmented mat-vec, bound by value at slot 3.
struct SegArgs {
    uint nblocks;   // input columns / the type's block values
    uint layer;     // the layer a raise is recorded in
    float eps;      // the norm's epsilon
    uint nseg;      // how many segments the dispatch runs
    MatvecSeg segs[BLOOMERY_MATVEC_MAX_SEGMENTS];
};

static_assert(sizeof(SegArgs) == 16 + 16 * BLOOMERY_MATVEC_MAX_SEGMENTS,
              "SegArgs is the header's four words and the segments' tables");

// The SIMD groups in a mat-vec's threadgroup: a function constant, so the
// count is a value the compiler knows (two rows per SIMD group, so a
// threadgroup of eight owns sixteen rows) and each count is its own pipeline.
// The host sets it on every mat-vec pipeline; a function that uses a function
// constant has no pipeline without it.
constant uint BLOOMERY_MATVEC_SIMD_GROUPS [[function_constant(0)]];

// The scale a norm multiplies the input by, from the sum of squares of the
// whole input.
inline float rms_scale(float sum_squares, uint count, float eps)
{
    return rsqrt(sum_squares / float(count) + eps);
}

// Whether a norm's sum of squares and scale are usable: a non-finite value in
// the input, or a scale that is not finite and positive (a zero mean with a
// zero epsilon, or an infinite epsilon), is undefined input.
inline bool rms_usable(float sum_squares, float scale)
{
    return isfinite(sum_squares) && isfinite(scale) && scale > 0.0f;
}

// The RMS norm of a chunk of rows: y = (x · gain) · scale, scale =
// 1 / sqrt(mean(x²) + eps), one threadgroup a row. The decode has no norm of
// its own to compare with: each mat-vec norms as it loads, in its own lane
// order. The router's is the one that keeps the gained input as an f32 vector
// (its threadgroup copy), so this kernel is the router's: the same threads
// (the router's SIMD-group count), the same float4 each thread takes
// (f4 = tid + threads · i, in i order), the four squares of a float4 added
// left to right, the SIMD-group sum, the groups' shares added in group order,
// and the same `rms_scale`. The gained value x · gain is rounded before the
// scale multiplies, as the router's dot sees it, so a row of this kernel is
// the router's normed input bit for bit.
//
// A row whose sum of squares or scale is unusable (`rms_usable`: a non-finite
// input, a zero mean with a zero epsilon, an infinite epsilon) raises the norm
// site and is NaN, and the other rows are unaffected. A value that is not
// finite beyond that (a gain that is not finite, a product past the range)
// raises the norm site too and stays what it is.
//
// Output: f32 rows. `norm_rows_value` is the one place a row's value is final;
// a narrower operand arm converts from it.
//
// The sandwich residual is this norm with a different store: norm_add_rows
// runs the same sums and adds the row's value into the residual stream, y +=
// add · ((x · gain) · scale), in place (x is the sub-layer's output, y the
// stream). Its sums are norm_rows' in the same order, so the value added is
// the bits norm_rows writes; the multiply by `add` and the add each round
// once. A row the norm cannot take poisons the stream's row with NaN, as it
// poisons the plain norm's output. norm_rows itself is not touched by it: its
// program is the plain norm's alone.

#define NORM_ROWS_SG BLOOMERY_ROUTER_SG
#define NORM_ROWS_THREADS (BLOOMERY_ROUTER_LANES * BLOOMERY_ROUTER_SG)

// The argument block, bound by value: the columns, the layer a raise is
// recorded in, the epsilon, and the strides in floats between rows of the
// input and of the output.
struct NormRowsArgs {
    uint cols;
    uint layer;
    float eps;
    uint x_stride;
    uint y_stride;
};

static_assert(sizeof(NormRowsArgs) == 20, "NormRowsArgs is 20 bytes: the host packs five words");

// The sandwich entry's argument block: the norm's, then the residual's scale.
struct NormAddArgs {
    NormRowsArgs norm;
    float add;
};

static_assert(sizeof(NormAddArgs) == 24, "NormAddArgs is 24 bytes: the host packs six words");

inline float4 norm_rows_load4(device const float *p, uint f4)
{
    return *((device const float4 *)(p + f4 * 4u));
}

inline float4 norm_rows_value(float4 x, float4 gain, float scale)
{
    return (x * gain) * scale;
}

// The norm site, raised by a row the norm cannot take.
inline void norm_rows_raise(device atomic_uint *word, uint layer)
{
    raise(word, layer, BLOOMERY_FAULT_SITE_NORM_QUANT);
}

kernel void norm_rows(device const float *x [[buffer(0)]],
                      device const float *gain [[buffer(1)]],
                      device float *y [[buffer(2)]],
                      constant NormRowsArgs &args [[buffer(3)]],
                      device atomic_uint *faults [[buffer(4)]],
                      uint row [[threadgroup_position_in_grid]],
                      uint tid [[thread_position_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]])
{
    threadgroup float tg_sums[NORM_ROWS_SG];
    device const float *xr = x + row * args.x_stride;
    device float *yr = y + row * args.y_stride;
    const uint f4s = args.cols / 4u;

    float sum_sq = 0.f;
    for (uint f4 = tid; f4 < f4s; f4 += NORM_ROWS_THREADS) {
        float4 xv = norm_rows_load4(xr, f4);
        sum_sq += xv.x * xv.x + xv.y * xv.y + xv.z * xv.z + xv.w * xv.w;
    }
    sum_sq = simd_sum(sum_sq);
    if (lane == 0u) {
        tg_sums[sg] = sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.f;
    for (uint g = 0u; g < NORM_ROWS_SG; ++g) {
        total += tg_sums[g];
    }
    const float scale = rms_scale(total, args.cols, args.eps);

    if (!rms_usable(total, scale)) {
        if (tid == 0u) {
            norm_rows_raise(faults, args.layer);
        }
        for (uint f4 = tid; f4 < f4s; f4 += NORM_ROWS_THREADS) {
            *((device float4 *)(yr + f4 * 4u)) = float4(NAN);
        }
        return;
    }
    bool bad = false;
    for (uint f4 = tid; f4 < f4s; f4 += NORM_ROWS_THREADS) {
        float4 value = norm_rows_value(norm_rows_load4(xr, f4), norm_rows_load4(gain, f4), scale);
        bad = bad || !all(isfinite(value));
        *((device float4 *)(yr + f4 * 4u)) = value;
    }
    if (bad) {
        norm_rows_raise(faults, args.layer);
    }
}

// The sandwich residual: the norm's row, added into the residual stream, y +=
// add · value, in place. The sums are norm_rows' in the same order (the same
// loads, the same simd_sum, the groups' shares in group order, the same
// `rms_scale`), so `value` is the bits norm_rows writes; the sandwich clause
// holds the two equal.
kernel void norm_add_rows(device const float *x [[buffer(0)]],
                          device const float *gain [[buffer(1)]],
                          device float *y [[buffer(2)]],
                          constant NormAddArgs &args [[buffer(3)]],
                          device atomic_uint *faults [[buffer(4)]],
                          uint row [[threadgroup_position_in_grid]],
                          uint tid [[thread_position_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]])
{
    threadgroup float tg_sums[NORM_ROWS_SG];
    device const float *xr = x + row * args.norm.x_stride;
    device float *yr = y + row * args.norm.y_stride;
    const uint f4s = args.norm.cols / 4u;

    float sum_sq = 0.f;
    for (uint f4 = tid; f4 < f4s; f4 += NORM_ROWS_THREADS) {
        float4 xv = norm_rows_load4(xr, f4);
        sum_sq += xv.x * xv.x + xv.y * xv.y + xv.z * xv.z + xv.w * xv.w;
    }
    sum_sq = simd_sum(sum_sq);
    if (lane == 0u) {
        tg_sums[sg] = sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.f;
    for (uint g = 0u; g < NORM_ROWS_SG; ++g) {
        total += tg_sums[g];
    }
    const float scale = rms_scale(total, args.norm.cols, args.norm.eps);

    if (!rms_usable(total, scale)) {
        if (tid == 0u) {
            norm_rows_raise(faults, args.norm.layer);
        }
        for (uint f4 = tid; f4 < f4s; f4 += NORM_ROWS_THREADS) {
            *((device float4 *)(yr + f4 * 4u)) = float4(NAN);
        }
        return;
    }
    bool bad = false;
    for (uint f4 = tid; f4 < f4s; f4 += NORM_ROWS_THREADS) {
        float4 value = norm_rows_value(norm_rows_load4(xr, f4), norm_rows_load4(gain, f4), scale);
        bad = bad || !all(isfinite(value));
        *((device float4 *)(yr + f4 * 4u)) = norm_rows_load4(yr, f4) + args.add * value;
    }
    if (bad) {
        norm_rows_raise(faults, args.norm.layer);
    }
}
