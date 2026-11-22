// The output head of Inkling 2 over f32 weights: the output norm, the
// projection of every kept vocabulary row, the soft cap, one dispatch of
// logits and the pick's own (`logits_argmax`, head.metal) after it.
//
// The input is the head mix's output (a hidden-wide row) and the output
// norm's gain; a threadgroup norms the row once into threadgroup memory
// (the sum of squares by fused multiply-adds ascending a thread's strided
// share, each SIMD group's lanes by simd_sum, the eight shares added in
// group order — the controller's own reduction, loop_control.metal), the
// normed value (x · scale) · gain rounding on its own. A SIMD group then
// owns a row: a lane's float4s of the row and the staged input, four fused
// multiply-adds each ascending, the lanes summed by simd_sum.
//
//   logit = cap · tanh(dot / cap)                  when the cap is on
//   logit = dot                                    otherwise
//
// The rows are the file's vocabulary cut to its kept prefix (the launch
// names the row count, a load-time view of the weights): a row past the
// count is not read, so a count that is no multiple of a threadgroup's
// eight rows takes a short last threadgroup.
//
// The staged row is the Mini's width (BLOOMERY_INK2_HIDDEN floats, 9 KiB);
// the host refuses another width by name.
//
// No silent failure: an input row whose sum of squares or scale is not
// usable (rms_usable, norm.metal) raises the norm site and every logit is
// NaN; a logit that is not finite is stored as it came and the pick raises
// the logit site, as every head's does.
//
// This file comes after fault.metal (raise), norm.metal (rms_scale,
// rms_usable) and head.metal.

#pragma METAL fp contract(off)

#define HF_THREADS 256u
#define HF_SG (HF_THREADS / 32u)
#define HF_K BLOOMERY_INK2_HIDDEN

static_assert(HF_K % 128u == 0u, "a float4 a lane a pass over the 32 lanes");

// The launch's argument block, bound by value.
struct HeadF32Args {
    uint rows;    // the kept vocabulary rows
    uint k;       // the hidden width, HF_K
    uint layer;
    float eps;
    float cap;    // the soft cap's value, read when capped
    uint capped;  // 1 when the cap is on
};

static_assert(sizeof(HeadF32Args) == 24, "HeadF32Args is 24 bytes: the host packs six words");

kernel void head_logits_f32(device const float *W [[buffer(0)]],
                            device const float *x [[buffer(1)]],
                            device const float *gain [[buffer(2)]],
                            device float *logits [[buffer(3)]],
                            constant HeadF32Args &args [[buffer(4)]],
                            device atomic_uint *faults [[buffer(5)]],
                            uint tg [[threadgroup_position_in_grid]],
                            uint tid [[thread_position_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]],
                            uint sg [[simdgroup_index_in_threadgroup]])
{
    threadgroup float staged[HF_K];
    threadgroup float partials[HF_SG];
    threadgroup float scale_of[1];

    float sq = 0.f;
    for (uint j = tid; j < HF_K; j += HF_THREADS) {
        sq = fma(x[j], x[j], sq);
    }
    sq = simd_sum(sq);
    if (lane == 0u) {
        partials[sg] = sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float total = 0.f;
        for (uint g = 0u; g < HF_SG; ++g) {
            total += partials[g];
        }
        const float scale = rms_scale(total, HF_K, args.eps);
        scale_of[0] = rms_usable(total, scale) ? scale : NAN;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float scale = scale_of[0];
    const bool usable = !isnan(scale);
    if (!usable && tid == 0u) {
        raise(faults, uint(args.layer), BLOOMERY_FAULT_SITE_NORM_QUANT);
    }
    for (uint j = tid; j < HF_K; j += HF_THREADS) {
        staged[j] = (x[j] * scale) * gain[j];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint row = tg * HF_SG + sg;
    if (row >= args.rows) {
        return;
    }
    if (!usable) {
        if (lane == 0u) {
            logits[row] = NAN;
        }
        return;
    }
    device const float4 *w = (device const float4 *)(W + (ulong)row * (ulong)HF_K);
    float acc = 0.f;
    for (uint i = 0u; i < HF_K / 128u; ++i) {
        const float4 wv = w[lane + 32u * i];
        const uint at = 4u * (lane + 32u * i);
        acc = fma(wv.x, staged[at], acc);
        acc = fma(wv.y, staged[at + 1u], acc);
        acc = fma(wv.z, staged[at + 2u], acc);
        acc = fma(wv.w, staged[at + 3u], acc);
    }
    const float dot = simd_sum(acc);
    if (lane == 0u) {
        logits[row] = args.capped != 0u ? args.cap * tanh(dot / args.cap) : dot;
    }
}
