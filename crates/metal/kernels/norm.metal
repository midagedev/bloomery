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
    uint  nblocks;  // input columns / 256
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
    uint nblocks;   // input columns / 256
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
