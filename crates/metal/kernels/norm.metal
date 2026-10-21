// What every mat-vec of the quantized kernels shares on the activation side:
// the argument block, and the RMS norm a mat-vec applies as it loads its
// input.
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
