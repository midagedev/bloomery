// The quantized mat-vec: y[r] = sum_c W[r, c] * x[c] for one f32 activation
// vector, one dispatch. The row loop is one template over the quant
// (quant.metal) and the modes; the entry points are the instantiations,
// under their host names, the names the host has always launched:
//
//   q4k_matvec                y = W . x
//   q4k_matvec_norm           y = W . rms_norm(x, gain, eps): the norm applied
//                             as x is loaded (norm.metal)
//   q4k_matvec_resid          y = y + W . x, in place: the residual epilogue
//   q4k_matvec_seg            the row segments of up to four tensors of one
//                             type and one width, in one dispatch
//   q4k_matvec_headnorm_resid y = y + W . (g * head_scale): the gated delta
//                             net's out projection, the head scale applied to
//                             each block's head groups
//   q6k_matvec / _norm / _resid / _seg   the same over Q6_K, instantiated in
//                             matvec_q6k.metal; Q6_K has no head-norm entry
//   a4_matvec / _norm / _resid / _seg /
//   a4_matvec_headnorm_resid  the same over MLX's affine 4-bit groups,
//                             instantiated in matvec_a4.metal
//   q8_0_matvec / ..., q5k_matvec / ...
//                             the same five over Q8_0 and Q5_K,
//                             instantiated in matvec_q8_0.metal and
//                             matvec_q5k.metal
//
// The row loop is llama.cpp's (kernel_mul_mv_q4_K_f32 and
// kernel_mul_mv_q6_K_f32, MIT licence; see THIRD_PARTY_NOTICES.md):
//   ggml/src/ggml-metal/kernels/mul_mv.metal
// The four-block and two-block walks, the row loop over a SIMD group's rows
// and the simd_sum reduction are its. The rows a SIMD group owns are the row
// loop's constant, four the landed count: the loop and the tail are written
// for any of 2, 4 and 8, and a SIMD group loads the input once for all its
// rows (the register reuse MLX's affine_qmv_fast is built on). The SIMD
// groups in a threadgroup are the function constant of norm.metal: 2 for the
// mat-vecs, the output head's logits among them, 8 for the head's argmax.
//
// What differs from llama.cpp's loop:
//   - the matrix is not padded: a SIMD group past the last row returns, and
//     each row's blocks are addressed from a pointer taken once before the
//     loop, clamped to the last row, so a row past the last re-reads the last
//     row (a defined access) and is never stored; llama.cpp advances one
//     pointer by a row's bytes and reads whole threadgroups of rows past the
//     end;
//   - the variants above.
//
// This file comes after norm.metal and quant.metal, and defines what the
// Q6_K, embed and head files use of it: the row loop and the entry templates.

#include <metal_stdlib>

using namespace metal;

#pragma METAL fp contract(off)

static_assert(BLOOMERY_MATVEC_ROWS_PER_SIMD_GROUP == 2 ||
              BLOOMERY_MATVEC_ROWS_PER_SIMD_GROUP == 4 ||
              BLOOMERY_MATVEC_ROWS_PER_SIMD_GROUP == 8,
              "the row loop and the tail are written for 2, 4 or 8 rows");

// The first row a SIMD group owns. `tg` is the threadgroup's index in the
// grid, or its index inside a segment.
inline int simd_first_row(uint tg, ushort sg) {
    return (int)((tg * BLOOMERY_MATVEC_SIMD_GROUPS + sg) * BLOOMERY_MATVEC_ROWS_PER_SIMD_GROUP);
}

// The rows of one segment or one tensor a threadgroup owns. The norm scales
// the whole input at the end, the head norm scales each block's two head
// groups once at the end of the block, and the residual adds into the
// output: the dot product is the same in all of them, so the rows they write
// are the same bits. ROW_RAISE: a norm-on-load row the scaled sum leaves
// not finite raises the norm site and is NaN. The head's logits turn it off:
// its argmax owns a non-finite logit and raises the logit site for it.
// SCALED: the input is already x · gain, written by the join before it
// (norm_add_next_kernel), and `gain` points at the one word of the scale that
// join computed in this quant's lane order; the load is the plain one and the
// rows are scaled and raised as the norm's, so they are the norm-on-load
// rows' bits.
template <typename Q, bool NORM, bool RESID, bool HEADNORM, bool ROW_RAISE = true,
          bool SCALED = false>
inline void matvec_rows(device const typename Q::Block * W,
                        device const float      * x,
                        device float            * y,
                        int nblocks,
                        float eps,
                        uint layer,
                        device const float      * gain,
                        threadgroup const float * scales,
                        device atomic_uint      * faults,
                        int first_row,
                        int rows,
                        ushort tiisg) {
    constexpr short NR0 = BLOOMERY_MATVEC_ROWS_PER_SIMD_GROUP;
    static_assert(!(NORM && SCALED), "a scaled input is normed once, by its join");

    const int nb = nblocks;
    if (first_row >= rows) {
        return;
    }
    // The join's scale, read before the walk so its load is in flight under it.
    const float in_scale = SCALED ? gain[0] : 1.f;
    // Each row's blocks, the last row's for a row past the last.
    device const typename Q::Block * rp[NR0];
    for (short r = 0; r < NR0; ++r) {
        rp[r] = W + (ulong)min(first_row + r, rows - 1) * (ulong)nb;
    }

    typename Q::Lane lane = Q::lane_of(tiisg);
    float sumf[NR0];
    for (short r = 0; r < NR0; ++r) sumf[r] = 0.f;
    float sum_sq = 0.f;

    // The lane's share of the input and the gain: the lane's first block's
    // base, the quant's load adding the lane's offset inside it. The gain is
    // bound by the norm entry only: the others form no pointer from it.
    device const float * xp = x + Q::first_block(lane) * Q::values_per_block();
    device const float * gp = NORM ? gain + Q::first_block(lane) * Q::values_per_block() : gain;

    for (int ib = Q::first_block(lane); ib < nb; ib += Q::block_stride()) {
        // The block's two heads: the head scale of each, once for the block.
        // The two are the quant's own: a 256-value block spans two heads, a
        // 64-value group sits inside one.
        float2 hs = {1.f, 1.f};
        if constexpr (HEADNORM) {
            hs = Q::head_scales(scales, ib);
        }
        if (NORM) {
            Q::template load<LOAD_NORM>(lane, xp, gp, 1.f, &sum_sq);
        } else {
            Q::template load<LOAD_PLAIN>(lane, xp, nullptr, 1.f, nullptr);
        }
        // A value the gated numerator cannot carry, or a group sum that left
        // the finite range: the gate's site is raised, and the sums it is in
        // are not finite either, so the rows are NaN.
        if constexpr (HEADNORM) {
            if (!Q::sums_finite(lane)) {
                raise(faults, layer, BLOOMERY_FAULT_SITE_LINEAR_GATE);
            }
        }

        for (short row = 0; row < NR0; ++row) {
            if constexpr (HEADNORM) {
                sumf[row] += Q::partial_head(rp[row][ib], lane, hs.x, hs.y);
            } else {
                sumf[row] += Q::partial(rp[row][ib], lane);
            }
        }

        xp += Q::block_stride() * Q::values_per_block();
        if (NORM) {
            gp += Q::block_stride() * Q::values_per_block();
        }
    }

    float scale = in_scale;
    if (NORM) {
        const float sum_squares = simd_sum(sum_sq);
        scale = rms_scale(sum_squares, (uint)nb * Q::values_per_block(), eps);
        if (!rms_usable(sum_squares, scale)) {
            if (tiisg == 0) {
                raise(faults, layer, BLOOMERY_FAULT_SITE_NORM_QUANT);
                for (short row = 0; row < NR0 && first_row + row < rows; ++row) {
                    y[first_row + row] = NAN;
                }
            }
            return;
        }
    }

    for (short row = 0; row < NR0 && first_row + row < rows; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (NORM || SCALED) {
            sum_all *= scale;
        }
        if (tiisg == 0) {
            // A row the scaled sum leaves not finite (a gain or a weight that is
            // not finite, or a scale its join found unusable) raises the norm
            // site and is NaN, not a plausible value.
            if ((NORM || SCALED) && ROW_RAISE && !isfinite(sum_all)) {
                raise(faults, layer, BLOOMERY_FAULT_SITE_NORM_QUANT);
                y[first_row + row] = NAN;
            } else if (RESID) {
                y[first_row + row] += sum_all;
            } else {
                y[first_row + row] = sum_all;
            }
        }
    }
}

// The plain mat-vec, storing or adding into the output.
template <typename Q, bool RESID>
kernel void matvec_kernel(device const typename Q::Block * W [[buffer(0)]],
                          device const float      * x [[buffer(1)]],
                          device float            * y [[buffer(2)]],
                          constant MatvecArgs     & args [[buffer(3)]],
                          uint3  tgpig [[threadgroup_position_in_grid]],
                          ushort tiisg [[thread_index_in_simdgroup]],
                          ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const int first_row = simd_first_row(tgpig.x, sgitg);
    matvec_rows<Q, false, RESID, false>(W, x, y, args.nblocks, args.eps, args.layer,
                                        nullptr, nullptr, nullptr, first_row, (int)args.rows, tiisg);
}

// The norm on load: the gain and the fault word are bound here.
template <typename Q>
kernel void matvec_norm_kernel(device const typename Q::Block * W [[buffer(0)]],
                               device const float      * x [[buffer(1)]],
                               device float            * y [[buffer(2)]],
                               constant MatvecArgs     & args [[buffer(3)]],
                               device const float      * gain [[buffer(4)]],
                               device atomic_uint      * faults [[buffer(5)]],
                               uint3  tgpig [[threadgroup_position_in_grid]],
                               ushort tiisg [[thread_index_in_simdgroup]],
                               ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const int first_row = simd_first_row(tgpig.x, sgitg);
    matvec_rows<Q, true, false, false>(W, x, y, args.nblocks, args.eps, args.layer,
                                       gain, nullptr, faults, first_row, (int)args.rows, tiisg);
}

// The row segments of up to four tensors of one type and one width: the
// threadgroup takes the segment whose first threadgroup it is, and the rows
// are the ones a launch of that segment alone would write. SCALED: the input
// is the join's x · gain and `gain` binds the join's scale word.
template <typename Q, bool SCALED = false>
kernel void matvec_seg_kernel(device const typename Q::Block * W [[buffer(0)]],
                              device const float      * x [[buffer(1)]],
                              device float            * y [[buffer(2)]],
                              constant SegArgs        & args [[buffer(3)]],
                              device const float      * gain [[buffer(4)]],
                              device atomic_uint      * faults [[buffer(5)]],
                              uint3  tgpig [[threadgroup_position_in_grid]],
                              ushort tiisg [[thread_index_in_simdgroup]],
                              ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    uint s = 0;
    for (uint i = 0; i < args.nseg; ++i) {
        if (tgpig.x >= args.segs[i].first_group) {
            s = i;
        }
    }
    const uint local = tgpig.x - args.segs[s].first_group;
    const int first_row = simd_first_row(local, sgitg);
    matvec_rows<Q, !SCALED, false, false, true, SCALED>(
        W + (ulong)args.segs[s].wblocks,
        x,
        y + args.segs[s].out,
        args.nblocks, args.eps, args.layer,
        gain, nullptr, faults,
        first_row, (int)args.segs[s].rows, tiisg);
}

// The gated delta net's out projection: the input is the gated numerator g
// times the norm scale of the head each column belongs to, and the scale comes
// from the head's partial sums of the squares of its values. The scales are
// computed once per threadgroup; the row loop reads g from device memory the
// way the plain residual path reads its input, and multiplies each block's two
// head groups by their scale once, so the per-value work the plain path does
// not do is two threadgroup loads and two products a block.
template <typename Q>
kernel void matvec_headnorm_kernel(device const typename Q::Block * W [[buffer(0)]],
                                    device const float      * g [[buffer(1)]],
                                    device const float      * partials [[buffer(2)]],
                                    device float            * y [[buffer(3)]],
                                    constant MatvecArgs     & args [[buffer(4)]],
                                    device atomic_uint      * faults [[buffer(5)]],
                                    uint3  tgpig [[threadgroup_position_in_grid]],
                                    ushort tiisg [[thread_index_in_simdgroup]],
                                    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    // The head scales, in threadgroup memory: a lane reads the scale of each
    // head its columns are in, and the lanes of a SIMD group do not read the
    // same head, so a shuffle cannot share them.
    threadgroup float scales[BLOOMERY_DELTA_VALUE_HEADS];
    // A lane a head, a further round for each SIMD width of heads.
    if (sgitg == 0) {
        for (uint head = tiisg; head < BLOOMERY_DELTA_VALUE_HEADS; head += 32u) {
            float sum = 0.f;
            for (short j = 0; j < BLOOMERY_DELTA_COLUMN_BLOCKS; ++j) {
                sum += partials[head * BLOOMERY_DELTA_COLUMN_BLOCKS + j];
            }
            const float mean = sum / float(BLOOMERY_DELTA_HEAD_COLS);
            const float s = rsqrt(mean + args.eps);
            if (!isfinite(sum) || !isfinite(s)) {
                // A head the norm cannot take is NaN, not a zero head: its values
                // are then not finite and the rows that read them are NaN.
                raise(faults, args.layer, BLOOMERY_FAULT_SITE_LINEAR_GATE);
                scales[head] = NAN;
            } else {
                scales[head] = s;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int first_row = simd_first_row(tgpig.x, sgitg);
    matvec_rows<Q, false, true, true>(W, g, y, args.nblocks, args.eps, args.layer,
                                      nullptr, scales, faults, first_row, (int)args.rows, tiisg);
}

// The next launch's norm, the second half of a sandwich join's work: the
// join of one row (norm.metal's norm_add_row), which also writes the stream's
// new row times the next launch's gain into `xn`, then the scale of that
// launch's norm into `xn[cols]`. The scale's sum is the next launch's own:
// SIMD group 0 walks the stream's row with quant Q's lane map and load, the
// walk matvec_rows and the pair run, so the scale is the bits that launch
// would compute, and its rows over `xn` (the SCALED entries) are its
// norm-on-load rows' bits. A sum or scale the norm cannot take raises the norm
// site in the next launch's layer, where that launch would raise it, and the
// scale is NaN, so every row it scales is NaN.
struct NextNormArgs {
    float eps;      // the next launch's norm epsilon
    uint layer;     // the layer the next launch's raise is recorded in
};

static_assert(sizeof(NextNormArgs) == 8, "NextNormArgs is 8 bytes: the host packs two words");

// The next launch's scale over the stream's row ROW of COLS values, into
// `xn[COLS]`: SIMD group 0's walk, in quant Q's lane order, the one sum both
// hand-ons write. A macro, not a function, so each kernel's program is the
// walk written in place. It reads `lane`, `next_gain`, `next`, `xn` and
// `faults` from the kernel's own names.
#define NEXT_NORM_SCALE(Q, ROW, COLS)                                                   \
    typename Q::Lane walk = Q::lane_of((ushort)lane);                                   \
    const int nb = (int)(COLS / (uint)Q::values_per_block());                           \
    float sum_sq = 0.f;                                                                 \
    device const float * yp = ROW + Q::first_block(walk) * Q::values_per_block();       \
    device const float * gp = next_gain + Q::first_block(walk) * Q::values_per_block(); \
    for (int ib = Q::first_block(walk); ib < nb; ib += Q::block_stride()) {             \
        Q::template load<LOAD_NORM>(walk, yp, gp, 1.f, &sum_sq);                        \
        yp += Q::block_stride() * Q::values_per_block();                                \
        gp += Q::block_stride() * Q::values_per_block();                                \
    }                                                                                   \
    const float sum_squares = simd_sum(sum_sq);                                         \
    const float scale = rms_scale(sum_squares, (uint)nb * Q::values_per_block(), next.eps); \
    if (lane == 0u) {                                                                   \
        if (rms_usable(sum_squares, scale)) {                                           \
            xn[COLS] = scale;                                                           \
        } else {                                                                        \
            raise(faults, next.layer, BLOOMERY_FAULT_SITE_NORM_QUANT);                  \
            xn[COLS] = NAN;                                                             \
        }                                                                               \
    }

template <typename Q>
kernel void norm_add_next_kernel(device const float *x [[buffer(0)]],
                                 device const float *gain [[buffer(1)]],
                                 device float *y [[buffer(2)]],
                                 constant NormAddArgs &args [[buffer(3)]],
                                 device atomic_uint *faults [[buffer(4)]],
                                 device const float *next_gain [[buffer(5)]],
                                 device float *xn [[buffer(6)]],
                                 constant NextNormArgs &next [[buffer(7)]],
                                 uint tid [[thread_position_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]],
                                 uint sg [[simdgroup_index_in_threadgroup]])
{
    threadgroup float tg_sums[NORM_ROWS_SG];
    norm_add_row(x, gain, y, args, faults, next_gain, xn, tid, lane, sg, tg_sums);
    // The row the walk reads is every thread's store.
    threadgroup_barrier(mem_flags::mem_device);
    if (sg != 0u) {
        return;
    }
    NEXT_NORM_SCALE(Q, y, args.norm.cols)
}

// A plain residual's hand-on: the stream the out projection already added
// into, read whole, times the next launch's gain into `xn`, each product
// rounded once (the product the next launch's norm on load forms), and the
// next launch's scale into `xn[cols]` (NEXT_NORM_SCALE). The stream is only
// read, so the walk needs no barrier after the stores. A stream the norm
// cannot take hands on a NaN scale and raises at the next launch's layer.
struct NormNextArgs {
    uint cols;          // the stream's values: whole blocks of Q
    NextNormArgs next;
};

static_assert(sizeof(NormNextArgs) == 12, "NormNextArgs is 12 bytes: the host packs three words");

template <typename Q>
kernel void norm_next_kernel(device const float *x [[buffer(0)]],
                             device const float *next_gain [[buffer(1)]],
                             device float *xn [[buffer(2)]],
                             constant NormNextArgs &args [[buffer(3)]],
                             device atomic_uint *faults [[buffer(4)]],
                             uint tid [[thread_position_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint sg [[simdgroup_index_in_threadgroup]])
{
    const uint f4s = args.cols / 4u;
    for (uint f4 = tid; f4 < f4s; f4 += NORM_ROWS_THREADS) {
        *((device float4 *)(xn + f4 * 4u)) = norm_rows_load4(x, f4) * norm_rows_load4(next_gain, f4);
    }
    if (sg != 0u) {
        return;
    }
    constant NextNormArgs &next = args.next;
    NEXT_NORM_SCALE(Q, x, args.cols)
}

// Each type's scaled segmented entry and the two hand-ons that feed it, the
// sandwich join's and the plain residual's: the entries a quant's file
// instantiates beside its own.
#define SCALED_SEG_ENTRIES(PREFIX, Q, BQ)                                              \
    template [[host_name(PREFIX "_matvec_seg_scaled")]]                                \
    kernel void matvec_seg_kernel<Q, true>(device const BQ * W [[buffer(0)]],          \
                                           device const float * x [[buffer(1)]],       \
                                           device float * y [[buffer(2)]],             \
                                           constant SegArgs & args [[buffer(3)]],      \
                                           device const float * gain [[buffer(4)]],    \
                                           device atomic_uint * faults [[buffer(5)]],  \
                                           uint3 tgpig [[threadgroup_position_in_grid]], \
                                           ushort tiisg [[thread_index_in_simdgroup]], \
                                           ushort sgitg [[simdgroup_index_in_threadgroup]]); \
    template [[host_name(PREFIX "_norm_add_next")]]                                    \
    kernel void norm_add_next_kernel<Q>(device const float * x [[buffer(0)]],          \
                                        device const float * gain [[buffer(1)]],       \
                                        device float * y [[buffer(2)]],                \
                                        constant NormAddArgs & args [[buffer(3)]],     \
                                        device atomic_uint * faults [[buffer(4)]],     \
                                        device const float * next_gain [[buffer(5)]],  \
                                        device float * xn [[buffer(6)]],               \
                                        constant NextNormArgs & next [[buffer(7)]],    \
                                        uint tid [[thread_position_in_threadgroup]],   \
                                        uint lane [[thread_index_in_simdgroup]],       \
                                        uint sg [[simdgroup_index_in_threadgroup]]);   \
    template [[host_name(PREFIX "_norm_next")]]                                        \
    kernel void norm_next_kernel<Q>(device const float * x [[buffer(0)]],              \
                                    device const float * next_gain [[buffer(1)]],      \
                                    device float * xn [[buffer(2)]],                   \
                                    constant NormNextArgs & args [[buffer(3)]],        \
                                    device atomic_uint * faults [[buffer(4)]],         \
                                    uint tid [[thread_position_in_threadgroup]],       \
                                    uint lane [[thread_index_in_simdgroup]],           \
                                    uint sg [[simdgroup_index_in_threadgroup]]);

// The Q4_K entries, under the names the host launches.

SCALED_SEG_ENTRIES("q4k", Q4K, block_q4_K)

template [[host_name("q4k_matvec")]]
kernel void matvec_kernel<Q4K, false>(device const block_q4_K * W [[buffer(0)]],
                                      device const float      * x [[buffer(1)]],
                                      device float            * y [[buffer(2)]],
                                      constant MatvecArgs     & args [[buffer(3)]],
                                      uint3  tgpig [[threadgroup_position_in_grid]],
                                      ushort tiisg [[thread_index_in_simdgroup]],
                                      ushort sgitg [[simdgroup_index_in_threadgroup]]);

template [[host_name("q4k_matvec_resid")]]
kernel void matvec_kernel<Q4K, true>(device const block_q4_K * W [[buffer(0)]],
                                      device const float      * x [[buffer(1)]],
                                      device float            * y [[buffer(2)]],
                                      constant MatvecArgs     & args [[buffer(3)]],
                                      uint3  tgpig [[threadgroup_position_in_grid]],
                                      ushort tiisg [[thread_index_in_simdgroup]],
                                      ushort sgitg [[simdgroup_index_in_threadgroup]]);

template [[host_name("q4k_matvec_norm")]]
kernel void matvec_norm_kernel<Q4K>(device const block_q4_K * W [[buffer(0)]],
                                    device const float      * x [[buffer(1)]],
                                    device float            * y [[buffer(2)]],
                                    constant MatvecArgs     & args [[buffer(3)]],
                                    device const float      * gain [[buffer(4)]],
                                    device atomic_uint      * faults [[buffer(5)]],
                                    uint3  tgpig [[threadgroup_position_in_grid]],
                                    ushort tiisg [[thread_index_in_simdgroup]],
                                    ushort sgitg [[simdgroup_index_in_threadgroup]]);

template [[host_name("q4k_matvec_seg")]]
kernel void matvec_seg_kernel<Q4K>(device const block_q4_K * W [[buffer(0)]],
                                    device const float      * x [[buffer(1)]],
                                    device float            * y [[buffer(2)]],
                                    constant SegArgs        & args [[buffer(3)]],
                                    device const float      * gain [[buffer(4)]],
                                    device atomic_uint      * faults [[buffer(5)]],
                                    uint3  tgpig [[threadgroup_position_in_grid]],
                                    ushort tiisg [[thread_index_in_simdgroup]],
                                    ushort sgitg [[simdgroup_index_in_threadgroup]]);

// The one entry the head-norm mode has: Q4_K's out projection.
template [[host_name("q4k_matvec_headnorm_resid")]]
kernel void matvec_headnorm_kernel<Q4K>(device const block_q4_K * W [[buffer(0)]],
                                        device const float      * g [[buffer(1)]],
                                        device const float      * partials [[buffer(2)]],
                                        device float            * y [[buffer(3)]],
                                        constant MatvecArgs     & args [[buffer(4)]],
                                        device atomic_uint      * faults [[buffer(5)]],
                                        uint3  tgpig [[threadgroup_position_in_grid]],
                                        ushort tiisg [[thread_index_in_simdgroup]],
                                        ushort sgitg [[simdgroup_index_in_threadgroup]]);
