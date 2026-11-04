// The M-row mat-vec family: one weight read multiplied into M = 2 activation
// rows, the shape a speculative verify of one proposed id needs — each weight
// row's block is loaded and unpacked once and dotted into both activation rows
// (Q::partial_rows), where running the 1-row kernel twice reads the weights
// twice and the prompt-rows GEMM reads a 64-row tile's worth. The kernels are
// written over M; the entries instantiate M = 2, under the names the host
// launches:
//
//   a4_matvec_resid_m2        y[m, r] += W[r, :] . x[m, :], in place
//   a4_matvec_seg_m2          the row segments of up to four tensors, the
//                             norm on the load, per activation row
//   a4_matvec_seg_scaled_m2, a4_pair_swiglu_scaled_m2,
//   a4_head_logits_scaled_m2  the segmented mat-vec, the gate-up pair with its
//                             SwiGLU epilogue (y[m, slot, o] = silu(gate_o . x[m])
//                             * (up_o . x[m])) and the head's logits (one row
//                             of logits an activation row; the pick is
//                             logits_argmax, one threadgroup an activation row)
//                             over inputs the join already normed: M rows of
//                             x . gain and M scale words (SCALED, the 1-row
//                             scaled entries' rule)
//   a4_matvec_headnorm_m2     the gated delta net's out projection: the
//                             head-normed residual mat-vec, each row's head
//                             scales computed from its own partial sums
//   a4_norm_next_m            the plain residual's hand-on over M rows: each
//                             row times the next launch's gain, and that
//                             launch's scale a row, the 1-row norm_next's bits
//
// THE ORDER IS THE GATE. Activation row m of any entry here is the 1-row
// kernel's bits on row m's input alone (matvec_q4k.metal's matvec_rows, its
// segmented and head entries and its head-normed out projection,
// experts.metal's pair): the lane map is the same
// Q::lane_of of the same thread index, each row's share of the input is
// loaded in the same block order (first_block + k . block_stride) into its
// own registers (Q::load_rows takes Q::load's operations per row), its squares
// join its own sum in that same order, its scale is rms_scale of the same
// simd_sum, its dot is accumulated per weight row by Q::partial_rows in
// partial's own value order, and the finished lane sums take the same
// simd_sum. The weights' read and their unpack (the code words, the masks, the
// widening) are the only thing the M rows share; the M argument blocks are the
// 1-row entries' own words plus the output stride between the M rows. SCALED:
// row m's input is the join's x . gain and `gain` binds the M scale words the
// join computed in this launch's own lane order; the load is the plain one and
// row m's dots are scaled by scale word m and raised as the norm's, the 1-row
// SCALED entries' own rule, row by row.
//
// Every loop over the weight rows and the activation rows is unrolled
// (FOR_UNROLL): a loop left rolled indexes the lane and sum arrays by its
// variable, which registers cannot take, and walks the weight rows one after
// the other, each read waiting on the one before; unrolled, all the weight
// rows' reads are issued before the first is used.
//
// The activations and the outputs are M rows at the width the launch names
// (the rows arena's own layout: a row of the buffer is the launch's width),
// the activations' stride the input columns and the outputs' the stride the
// entry names. A row whose norm cannot take its input raises the norm site
// and is NaN in its own rows only; the other rows' writes are unchanged.
//
// This file comes after norm.metal, quant.metal, gemm_shared.metal
// (pair_swiglu), matvec_q4k.metal (simd_first_row) and experts.metal
// (PairArgs, pair_expert), and instantiates the A4 entries only:
// the decode owner's partial_rows is A4's, the kind an MLX 4-bit folder of a
// Bf16 scale pair opens as. Another quant gains the family by gaining
// partial_rows and instantiating the same bodies.

#include <metal_stdlib>

using namespace metal;

#pragma METAL fp contract(off)

// The plain M-row entries' argument block, bound by value at slot 3 (the
// head-normed entry's at slot 4): the 1-row MatvecArgs' four words and the
// output stride between the M rows.
struct MatvecMArgs {
    uint  rows;     // output rows an activation row takes
    uint  nblocks;  // input columns / the type's block values
    uint  layer;    // the layer a raise is recorded in
    float eps;      // the norm's epsilon (read by the norm and head-norm variants)
    uint  stride;   // floats from one activation row's outputs to the next
};

static_assert(sizeof(MatvecMArgs) == 20, "MatvecMArgs is five words");

// The segmented M-row entry's, bound by value at slot 3: the 1-row SegArgs'
// header plus the same output stride, the segments' tables unchanged.
struct SegMArgs {
    uint  nblocks;
    uint  layer;
    float eps;
    uint  nseg;
    uint  stride;   // floats from one activation row's outputs to the next
    MatvecSeg segs[BLOOMERY_MATVEC_MAX_SEGMENTS];
};

static_assert(sizeof(SegMArgs) == 20 + 16 * BLOOMERY_MATVEC_MAX_SEGMENTS,
              "SegMArgs is the header's five words and the segments' tables");

// The weight rows a SIMD group owns in the M-row entries: the geometry's
// count, as in the 1-row entries (simd_first_row names the first), and the
// host launches the same count. More activation rows than the family's two
// would hold more live input values a lane, and each weight row it owns needs
// registers too (a spilled register is a device-memory round trip every
// block). The pair's gate rows a SIMD group owns are half of them, each with
// its up row.
constexpr short mrow_nr0() {
    return BLOOMERY_MATVEC_ROWS_PER_SIMD_GROUP;
}

// The M-row row loop: matvec_rows' walk over the weight rows a SIMD group
// owns, every activation row loaded by the same lane map into its own
// registers and every weight row's block dotted into all of them. Every loop
// here is unrolled: the weight rows' reads are issued together before the
// first is used, and no register array is indexed by a loop variable.
template <typename Q, short M, bool NORM, bool RESID, bool ROW_RAISE = true,
          bool SCALED = false>
inline void matvec_rows_m(device const typename Q::Block * W,
                          device const float      * x,
                          device float            * y,
                          int nblocks,
                          float eps,
                          uint layer,
                          device const float      * gain,
                          device atomic_uint      * faults,
                          int first_row,
                          int rows,
                          uint out_stride,
                          ushort tiisg) {
    constexpr short NR0 = mrow_nr0();
    static_assert(!(NORM && SCALED), "a scaled input is normed once, by its join");
    const int nb = nblocks;
    if (first_row >= rows) {
        return;
    }
    // The join's scale of each row, read before the walk so its loads are in
    // flight under it.
    float scale[M];
    FOR_UNROLL (short m = 0; m < M; ++m) {
        scale[m] = SCALED ? gain[m] : 1.f;
    }
    // Each weight row's blocks, the last row's for a row past the last: the
    // 1-row loop's clamp.
    device const typename Q::Block * rp[NR0];
    FOR_UNROLL (short r = 0; r < NR0; ++r) {
        rp[r] = W + (ulong)min(first_row + r, rows - 1) * (ulong)nb;
    }

    const typename Q::Lane lane = Q::lane_of(tiisg);
    typename Q::Lane lanes[M];
    FOR_UNROLL (short m = 0; m < M; ++m) {
        lanes[m] = lane;
    }
    float sumf[NR0][M];
    FOR_UNROLL (short r = 0; r < NR0; ++r) {
        FOR_UNROLL (short m = 0; m < M; ++m) {
            sumf[r][m] = 0.f;
        }
    }
    float sum_sq[M];
    FOR_UNROLL (short m = 0; m < M; ++m) {
        sum_sq[m] = 0.f;
    }

    // Activation row m's lane share of the input is row 0's, its row span on;
    // the gain is every row's, shared.
    const long row_span = (long)nb * (long)Q::values_per_block();
    device const float * xp = x + Q::first_block(lane) * Q::values_per_block();
    device const float * gp = NORM ? gain + Q::first_block(lane) * Q::values_per_block() : gain;

    for (int ib = Q::first_block(lane); ib < nb; ib += Q::block_stride()) {
        typename Q::Words wd[NR0];
        FOR_UNROLL (short row = 0; row < NR0; ++row) {
            wd[row] = Q::words_of(rp[row][ib], lane);
        }
        Q::template load_rows<NORM ? LOAD_NORM : LOAD_PLAIN, M>(lanes, xp, row_span, gp, sum_sq);
        FOR_UNROLL (short row = 0; row < NR0; ++row) {
            Q::template partial_rows<M>(wd[row], lanes, sumf[row]);
        }
        xp += Q::block_stride() * Q::values_per_block();
        if (NORM) {
            gp += Q::block_stride() * Q::values_per_block();
        }
    }

    // Each row's own scale of its own sum; a row the norm cannot take is NaN
    // in its own outputs and raises where the 1-row kernel raises, and the
    // other rows still write theirs.
    if (NORM) {
        FOR_UNROLL (short m = 0; m < M; ++m) {
            const float sum_squares = simd_sum(sum_sq[m]);
            scale[m] = rms_scale(sum_squares, (uint)nb * Q::values_per_block(), eps);
            if (!rms_usable(sum_squares, scale[m])) {
                if (tiisg == 0) {
                    raise(faults, layer, BLOOMERY_FAULT_SITE_NORM_QUANT);
                    for (short row = 0; row < NR0 && first_row + row < rows; ++row) {
                        y[(ulong)m * out_stride + (uint)(first_row + row)] = NAN;
                    }
                }
                scale[m] = NAN;
            }
        }
    }

    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        if (first_row + row >= rows) {
            break;
        }
        FOR_UNROLL (short m = 0; m < M; ++m) {
            if (NORM && isnan(scale[m])) {
                continue;
            }
            float sum_all = simd_sum(sumf[row][m]);
            if (NORM || SCALED) {
                sum_all *= scale[m];
            }
            if (tiisg == 0) {
                // A row the scaled sum leaves not finite (a gain or a weight
                // that is not finite, or a scale its join found unusable)
                // raises the norm site and is NaN, not a plausible value; the
                // head's logits turn that off, its argmax owning a non-finite
                // logit.
                if ((NORM || SCALED) && ROW_RAISE && !isfinite(sum_all)) {
                    raise(faults, layer, BLOOMERY_FAULT_SITE_NORM_QUANT);
                    y[(ulong)m * out_stride + (uint)(first_row + row)] = NAN;
                } else if (RESID) {
                    y[(ulong)m * out_stride + (uint)(first_row + row)] += sum_all;
                } else {
                    y[(ulong)m * out_stride + (uint)(first_row + row)] = sum_all;
                }
            }
        }
    }
}

// The plain M-row mat-vec, adding into each row's outputs.
template <typename Q, short M>
kernel void matvec_m_kernel(device const typename Q::Block * W [[buffer(0)]],
                            device const float      * x [[buffer(1)]],
                            device float            * y [[buffer(2)]],
                            constant MatvecMArgs    & args [[buffer(3)]],
                            uint3  tgpig [[threadgroup_position_in_grid]],
                            ushort tiisg [[thread_index_in_simdgroup]],
                            ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const int first_row = simd_first_row(tgpig.x, sgitg);
    matvec_rows_m<Q, M, false, true, true>(W, x, y, (int)args.nblocks, args.eps, args.layer,
                                           nullptr, nullptr, first_row, (int)args.rows,
                                           args.stride, tiisg);
}

// The row segments of up to four tensors of one type and one width, the norm
// on the load, per activation row: the segment a threadgroup takes is the
// 1-row segmented entry's own rule, and each row's outputs sit at the
// segment's output offset inside that row's own stride.
template <typename Q, short M, bool SCALED = false>
kernel void matvec_seg_m_kernel(device const typename Q::Block * W [[buffer(0)]],
                                device const float      * x [[buffer(1)]],
                                device float            * y [[buffer(2)]],
                                constant SegMArgs       & args [[buffer(3)]],
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
    matvec_rows_m<Q, M, !SCALED, false, true, SCALED>(W + (ulong)args.segs[s].wblocks, x,
                                                      y + args.segs[s].out,
                                                      (int)args.nblocks, args.eps, args.layer,
                                                      gain, faults, first_row,
                                                      (int)args.segs[s].rows, args.stride, tiisg);
}

// The head's logits over the join's normed rows, one row of logits an
// activation row (matvec_rows_m's SCALED). The pick over each row is the
// one-threadgroup logits_argmax, launched by the host once an activation row
// over that row's logits.
template <typename Q, short M>
kernel void head_logits_m_kernel(device const typename Q::Block *weights [[buffer(0)]],
                                 device const float *x [[buffer(1)]],
                                 device const float *gain [[buffer(2)]],
                                 device float *logits [[buffer(3)]],
                                 device atomic_uint *faults [[buffer(4)]],
                                 constant MatvecMArgs &args [[buffer(5)]],
                                 uint tg [[threadgroup_position_in_grid]],
                                 ushort tiisg [[thread_index_in_simdgroup]],
                                 ushort sg [[simdgroup_index_in_threadgroup]]) {
    const int first_row = simd_first_row(tg, sg);
    matvec_rows_m<Q, M, false, false, false, true>(weights, x, logits, (int)args.nblocks,
                                                   args.eps, args.layer, gain, faults,
                                                   first_row, (int)args.rows, args.stride, tiisg);
}

// The gated delta net's out projection over M activation rows: the 1-row
// head-normed residual mat-vec (matvec_q4k.metal's matvec_headnorm_kernel)
// with the weights' read and their unpack the only thing the M rows share.
// Each row's head scales are computed once per threadgroup into threadgroup
// memory, SIMD group 0 running the 1-row kernel's loop over that row's
// partial sums before the one barrier, and the row loop below scales each
// block's finished contribution by its row's head scale, the 1-row
// partial_head's own rule. The argument block is the family's MatvecMArgs:
// the numerators' and the partials' row strides the kernel derives, a
// numerator row the launch's columns and a partials row the heads' own
// table.
//
// THE ORDER IS THE GATE. Activation row m of this entry is the 1-row kernel's
// bits on row m's inputs alone: the lane map is the same Q::lane_of of the
// same thread index, the row's share of its numerator is loaded in the same
// block order (first_block + k . block_stride) into its own registers, its
// head scales are the 1-row loop's own sum order (j = 0 .. column blocks) and
// its rsqrt on its own partial sums, its dot is accumulated per weight row by
// Q::partial_rows in partial's own value order with the head scale applied to
// the finished block contribution (Q::partial_head_rows, partial_head's
// rule), and the finished lane sums take the same simd_sum.
//
// A head whose sum or scale is not finite raises the linear-gate site and is
// NaN in that row only; a value a row's numerator cannot carry, or a group
// sum that left the finite range, raises the same site (Q::sums_finite, the
// 1-row arm's own ask), and the other row's writes are its exact bits.
template <typename Q, short M>
inline void matvec_headnorm_rows_m(device const typename Q::Block * W,
                                   device const float      * g,
                                   device float            * y,
                                   threadgroup const float * scales,
                                   int nblocks,
                                   uint layer,
                                   device atomic_uint      * faults,
                                   int first_row,
                                   int rows,
                                   uint out_stride,
                                   ushort tiisg) {
    constexpr short NR0 = mrow_nr0();
    const int nb = nblocks;
    if (first_row >= rows) {
        return;
    }
    // Each row's blocks, the last row's for a row past the last: the 1-row
    // loop's clamp.
    device const typename Q::Block * rp[NR0];
    FOR_UNROLL (short r = 0; r < NR0; ++r) {
        rp[r] = W + (ulong)min(first_row + r, rows - 1) * (ulong)nb;
    }

    const typename Q::Lane lane = Q::lane_of(tiisg);
    typename Q::Lane lanes[M];
    FOR_UNROLL (short m = 0; m < M; ++m) {
        lanes[m] = lane;
    }
    float sumf[NR0][M];
    FOR_UNROLL (short r = 0; r < NR0; ++r) {
        FOR_UNROLL (short m = 0; m < M; ++m) {
            sumf[r][m] = 0.f;
        }
    }

    // Activation row m's lane share of the numerator is row 0's, its row
    // span on.
    const long row_span = (long)nb * (long)Q::values_per_block();
    device const float * gp = g + Q::first_block(lane) * Q::values_per_block();

    for (int ib = Q::first_block(lane); ib < nb; ib += Q::block_stride()) {
        typename Q::Words wd[NR0];
        FOR_UNROLL (short row = 0; row < NR0; ++row) {
            wd[row] = Q::words_of(rp[row][ib], lane);
        }
        Q::template load_rows<LOAD_PLAIN, M>(lanes, gp, row_span, nullptr, nullptr);
        // A value a row's numerator cannot carry, or a group sum that left
        // the finite range: the gate's site is raised, and the sums it is in
        // are not finite either, so the rows are NaN.
        FOR_UNROLL (short m = 0; m < M; ++m) {
            if (!Q::sums_finite(lanes[m])) {
                raise(faults, layer, BLOOMERY_FAULT_SITE_LINEAR_GATE);
            }
        }
        // The block's head scale of each row, the 1-row arm's own ask.
        float lo[M];
        FOR_UNROLL (short m = 0; m < M; ++m) {
            lo[m] = Q::head_scales(scales + m * BLOOMERY_DELTA_VALUE_HEADS, ib).x;
        }
        FOR_UNROLL (short row = 0; row < NR0; ++row) {
            Q::template partial_head_rows<M>(wd[row], lanes, lo, sumf[row]);
        }
        gp += Q::block_stride() * Q::values_per_block();
    }

    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        if (first_row + row >= rows) {
            break;
        }
        FOR_UNROLL (short m = 0; m < M; ++m) {
            const float sum_all = simd_sum(sumf[row][m]);
            if (tiisg == 0) {
                y[(ulong)m * out_stride + (uint)(first_row + row)] += sum_all;
            }
        }
    }
}

// The head-normed M-row entry: the scales of the M rows in threadgroup
// memory, row m's at `m * heads`, SIMD group 0 filling them by the 1-row
// kernel's loop before the one barrier.
template <typename Q, short M>
kernel void matvec_headnorm_m_kernel(device const typename Q::Block * W [[buffer(0)]],
                                     device const float      * g [[buffer(1)]],
                                     device const float      * partials [[buffer(2)]],
                                     device float            * y [[buffer(3)]],
                                     constant MatvecMArgs    & args [[buffer(4)]],
                                     device atomic_uint      * faults [[buffer(5)]],
                                     uint3  tgpig [[threadgroup_position_in_grid]],
                                     ushort tiisg [[thread_index_in_simdgroup]],
                                     ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float scales[M * BLOOMERY_DELTA_VALUE_HEADS];
    // A lane a head, a further round for each SIMD width of heads: the 1-row
    // kernel's loop, a row's partial sums at a time. A partials row is the
    // heads' own table, the instance's delta geometry.
    const uint pstride = BLOOMERY_DELTA_VALUE_HEADS * BLOOMERY_DELTA_COLUMN_BLOCKS;
    if (sgitg == 0) {
        for (uint head = tiisg; head < BLOOMERY_DELTA_VALUE_HEADS; head += 32u) {
            FOR_UNROLL (short m = 0; m < M; ++m) {
                float sum = 0.f;
                for (short j = 0; j < BLOOMERY_DELTA_COLUMN_BLOCKS; ++j) {
                    sum += partials[(ulong)m * pstride + head * BLOOMERY_DELTA_COLUMN_BLOCKS + j];
                }
                const float mean = sum / float(BLOOMERY_DELTA_HEAD_COLS);
                const float s = rsqrt(mean + args.eps);
                if (!isfinite(sum) || !isfinite(s)) {
                    // A head the norm cannot take is NaN, not a zero head:
                    // its values are then not finite and the rows that read
                    // them are NaN.
                    raise(faults, args.layer, BLOOMERY_FAULT_SITE_LINEAR_GATE);
                    scales[m * BLOOMERY_DELTA_VALUE_HEADS + head] = NAN;
                } else {
                    scales[m * BLOOMERY_DELTA_VALUE_HEADS + head] = s;
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    matvec_headnorm_rows_m<Q, M>(W, g, y, scales, (int)args.nblocks, args.layer, faults,
                                 simd_first_row(tgpig.x, sgitg), (int)args.rows, args.stride,
                                 tiisg);
}

// The M-row gate-up pair: pair_slot's (experts.metal) walk with M activation
// rows. Every slot reads the same M rows and writes its outputs inside each
// row's own span, the slots' whole width apart: the y the session holds is M
// rows of the instance's slots. A SIMD group owns half the weight rows' count
// of gate rows, each with its up row. `x` is the join's x . gain rows and
// `gain` binds its M scale words (matvec_rows_m's SCALED): the load is the
// plain one and the dots are scaled as the norm's, row m's bits the 1-row
// scaled pair's.
template <typename Q, short M>
inline void pair_slot_m(device const typename Q::Block * gate,
                        device const typename Q::Block * up,
                        device const float * x,
                        device const float * gain,
                        device float * y,
                        constant PairArgs & args,
                        uint nblocks,
                        uint slot,
                        int row0,
                        ushort tiisg) {
    constexpr short OUTS = mrow_nr0() / 2;
    const typename Q::Lane lane = Q::lane_of(tiisg);
    typename Q::Lane lanes[M];
    FOR_UNROLL (short m = 0; m < M; ++m) {
        lanes[m] = lane;
    }
    float sumg[OUTS][M];
    float sumu[OUTS][M];
    FOR_UNROLL (short o = 0; o < OUTS; ++o) {
        FOR_UNROLL (short m = 0; m < M; ++m) {
            sumg[o][m] = 0.f;
            sumu[o][m] = 0.f;
        }
    }
    // The join's scale of each row, read before the walk so its loads are in
    // flight under it.
    float in_scale[M];
    FOR_UNROLL (short m = 0; m < M; ++m) {
        in_scale[m] = gain[m];
    }
    device const typename Q::Block * grow[OUTS];
    device const typename Q::Block * urow[OUTS];
    FOR_UNROLL (short o = 0; o < OUTS; ++o) {
        grow[o] = gate + (ulong)(row0 + o) * (ulong)nblocks;
        urow[o] = up + (ulong)(row0 + o) * (ulong)nblocks;
    }
    const long row_span = (long)nblocks * (long)Q::values_per_block();
    device const float * xp = x + Q::first_block(lane) * Q::values_per_block();

    for (int ib = Q::first_block(lane); ib < (int)nblocks; ib += Q::block_stride()) {
        typename Q::Words wg[OUTS];
        typename Q::Words wu[OUTS];
        FOR_UNROLL (short o = 0; o < OUTS; ++o) {
            wg[o] = Q::words_of(grow[o][ib], lane);
            wu[o] = Q::words_of(urow[o][ib], lane);
        }
        Q::template load_rows<LOAD_PLAIN, M>(lanes, xp, row_span, nullptr, nullptr);
        FOR_UNROLL (short o = 0; o < OUTS; ++o) {
            Q::template partial_rows<M>(wg[o], lanes, sumg[o]);
            Q::template partial_rows<M>(wu[o], lanes, sumu[o]);
        }
        xp += Q::block_stride() * Q::values_per_block();
    }

    const ulong pitch = (ulong)BLOOMERY_EXPERT_SLOTS * args.stride;
    FOR_UNROLL (short m = 0; m < M; ++m) {
        const float scale = in_scale[m];
        FOR_UNROLL (short o = 0; o < OUTS; ++o) {
            if (row0 + o >= (int)args.rows) {
                break;
            }
            const float g = simd_sum(sumg[o][m]) * scale;
            const float u = simd_sum(sumu[o][m]) * scale;
            if (tiisg == 0u) {
                y[pitch * (ulong)m + (ulong)slot * args.stride + (uint)(row0 + o)] =
                    pair_swiglu(g, u);
            }
        }
    }
}

// The M-row pair over one type's stacks and shared expert (a shared expert of
// another type has no M-row entry): the 1-row pair's slot rule whole, every
// slot an activation row set.
template <typename Q, short M>
kernel void pair_swiglu_m_kernel(device const typename Q::Block * gate [[buffer(0)]],
                                 device const typename Q::Block * up [[buffer(1)]],
                                 device const typename Q::Block * shexp_gate [[buffer(2)]],
                                 device const typename Q::Block * shexp_up [[buffer(3)]],
                                 device const float * x [[buffer(4)]],
                                 device const float * gain [[buffer(5)]],
                                 device const int * ids [[buffer(6)]],
                                 device float * y [[buffer(7)]],
                                 device atomic_uint * faults [[buffer(8)]],
                                 constant PairArgs & args [[buffer(9)]],
                                 uint3 tgpig [[threadgroup_position_in_grid]],
                                 ushort tiisg [[thread_index_in_simdgroup]],
                                 ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint slot = tgpig.y;
    if (slot >= BLOOMERY_EXPERT_SLOTS) {
        return;
    }
    const int row0 = (int)((tgpig.x * BLOOMERY_MATVEC_SIMD_GROUPS + (uint)sgitg) *
                           (uint)(mrow_nr0() / 2));
    if (row0 >= (int)args.rows) {
        return;
    }

    int expert = 0;
    if (pair_expert(ids, slot, args.n_experts, expert)) {
        if (tiisg == 0u) {
            const ulong pitch = (ulong)BLOOMERY_EXPERT_SLOTS * args.stride;
            raise(faults, args.layer, BLOOMERY_FAULT_SITE_EXPERT_ID);
            FOR_UNROLL (short m = 0; m < M; ++m) {
                FOR_UNROLL (short o = 0; o < mrow_nr0() / 2 && row0 + o < (int)args.rows; ++o) {
                    y[pitch * (ulong)m + (ulong)slot * args.stride + (uint)(row0 + o)] = NAN;
                }
            }
        }
        return;
    }
    if (slot == BLOOMERY_EXPERT_SHARED) {
        pair_slot_m<Q, M>(shexp_gate, shexp_up, x, gain, y, args, args.nblocks, slot, row0,
                          tiisg);
    } else {
        const ulong expert_blocks = (ulong)expert * (ulong)args.rows * (ulong)args.nblocks;
        pair_slot_m<Q, M>(gate + expert_blocks, up + expert_blocks, x, gain, y, args,
                          args.nblocks, slot, row0, tiisg);
    }
}

// The A4 entries, under the names the host launches, M = 2.
#define MROW_MATVEC_ENTRY(NAME, Q, BQ, M)                                               \
    template [[host_name(NAME)]]                                                        \
    kernel void matvec_m_kernel<Q, M>(device const BQ * W [[buffer(0)]],                \
                                      device const float * x [[buffer(1)]],             \
                                      device float * y [[buffer(2)]],                   \
                                      constant MatvecMArgs & args [[buffer(3)]],        \
                                      uint3 tgpig [[threadgroup_position_in_grid]],     \
                                      ushort tiisg [[thread_index_in_simdgroup]],       \
                                      ushort sgitg [[simdgroup_index_in_threadgroup]]);

#define MROW_SEG_ENTRY(NAME, Q, BQ, M, SCALED)                                          \
    template [[host_name(NAME)]]                                                        \
    kernel void matvec_seg_m_kernel<Q, M, SCALED>(device const BQ * W [[buffer(0)]],            \
                                          device const float * x [[buffer(1)]],         \
                                          device float * y [[buffer(2)]],               \
                                          constant SegMArgs & args [[buffer(3)]],       \
                                          device const float * gain [[buffer(4)]],      \
                                          device atomic_uint * faults [[buffer(5)]],    \
                                          uint3 tgpig [[threadgroup_position_in_grid]], \
                                          ushort tiisg [[thread_index_in_simdgroup]],   \
                                          ushort sgitg [[simdgroup_index_in_threadgroup]]);

#define MROW_HEAD_ENTRY(NAME, Q, BQ, M)                                                 \
    template [[host_name(NAME)]]                                                        \
    kernel void head_logits_m_kernel<Q, M>(device const BQ *weights [[buffer(0)]],      \
                                           device const float *x [[buffer(1)]],          \
                                           device const float *gain [[buffer(2)]],       \
                                           device float *logits [[buffer(3)]],           \
                                           device atomic_uint *faults [[buffer(4)]],     \
                                           constant MatvecMArgs &args [[buffer(5)]],     \
                                           uint tg [[threadgroup_position_in_grid]],    \
                                           ushort tiisg [[thread_index_in_simdgroup]],  \
                                           ushort sg [[simdgroup_index_in_threadgroup]]);

#define MROW_PAIR_ENTRY(NAME, Q, BQ, M)                                                 \
    template [[host_name(NAME)]]                                                        \
    kernel void pair_swiglu_m_kernel<Q, M>(device const BQ * gate [[buffer(0)]],        \
                                           device const BQ * up [[buffer(1)]],           \
                                           device const BQ * shexp_gate [[buffer(2)]],   \
                                           device const BQ * shexp_up [[buffer(3)]],     \
                                           device const float * x [[buffer(4)]],         \
                                           device const float * gain [[buffer(5)]],      \
                                           device const int * ids [[buffer(6)]],         \
                                           device float * y [[buffer(7)]],               \
                                           device atomic_uint * faults [[buffer(8)]],    \
                                           constant PairArgs & args [[buffer(9)]],       \
                                           uint3 tgpig [[threadgroup_position_in_grid]], \
                                           ushort tiisg [[thread_index_in_simdgroup]],   \
                                           ushort sgitg [[simdgroup_index_in_threadgroup]]);

#define MROW_HEADNORM_ENTRY(NAME, Q, BQ, M)                                             \
    template [[host_name(NAME)]]                                                        \
    kernel void matvec_headnorm_m_kernel<Q, M>(device const BQ * W [[buffer(0)]],        \
                                               device const float * g [[buffer(1)]],      \
                                               device const float * partials [[buffer(2)]],\
                                               device float * y [[buffer(3)]],            \
                                               constant MatvecMArgs & args [[buffer(4)]], \
                                               device atomic_uint * faults [[buffer(5)]], \
                                               uint3 tgpig [[threadgroup_position_in_grid]],\
                                               ushort tiisg [[thread_index_in_simdgroup]],\
                                               ushort sgitg [[simdgroup_index_in_threadgroup]]);

MROW_MATVEC_ENTRY("a4_matvec_resid_m2", A4, block_mlx_a4, 2)
MROW_SEG_ENTRY("a4_matvec_seg_m2", A4, block_mlx_a4, 2, false)
MROW_SEG_ENTRY("a4_matvec_seg_scaled_m2", A4, block_mlx_a4, 2, true)
MROW_PAIR_ENTRY("a4_pair_swiglu_scaled_m2", A4, block_mlx_a4, 2)
MROW_HEAD_ENTRY("a4_head_logits_scaled_m2", A4, block_mlx_a4, 2)
MROW_HEADNORM_ENTRY("a4_matvec_headnorm_m2", A4, block_mlx_a4, 2)

// The plain residual's hand-on over M rows (norm_next_kernel's own body, one
// threadgroup a row): the stream's row r, which the out projection already
// added into, times the next launch's gain into `xn` row r, and that launch's
// scale of row r into the word after the M rows, `xn[M * cols + r]`. A row is
// the 1-row hand-on's whole program on that row alone: the products, the walk
// in the next launch's lane order, the usable test. A row the norm cannot take
// raises the norm site in the next launch's layer, where that launch would,
// and its scale is NaN, so every row of it that a reader scales is NaN; the
// other rows' words are unchanged.
struct NormNextMArgs {
    uint cols;          // the stream's values: whole blocks of Q
    uint rows;          // the M rows
    NextNormArgs next;
};

static_assert(sizeof(NormNextMArgs) == 16, "NormNextMArgs is 16 bytes: the host packs four words");

template <typename Q>
kernel void norm_next_m_kernel(device const float *x [[buffer(0)]],
                               device const float *next_gain [[buffer(1)]],
                               device float *xn [[buffer(2)]],
                               constant NormNextMArgs &args [[buffer(3)]],
                               device atomic_uint *faults [[buffer(4)]],
                               uint tid [[thread_position_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]],
                               uint sg [[simdgroup_index_in_threadgroup]],
                               uint row [[threadgroup_position_in_grid]])
{
    device const float *x_row = x + (ulong)row * args.cols;
    device float *xn_row = xn + (ulong)row * args.cols;
    const uint f4s = args.cols / 4u;
    for (uint f4 = tid; f4 < f4s; f4 += NORM_ROWS_THREADS) {
        *((device float4 *)(xn_row + f4 * 4u)) =
            norm_rows_load4(x_row, f4) * norm_rows_load4(next_gain, f4);
    }
    if (sg != 0u) {
        return;
    }
    constant NextNormArgs &next = args.next;
    NEXT_NORM_SCALE(Q, x_row, args.cols, (ulong)args.rows * args.cols + row)
}

template [[host_name("a4_norm_next_m")]]
kernel void norm_next_m_kernel<A4>(device const float *x [[buffer(0)]],
                                   device const float *next_gain [[buffer(1)]],
                                   device float *xn [[buffer(2)]],
                                   constant NormNextMArgs &args [[buffer(3)]],
                                   device atomic_uint *faults [[buffer(4)]],
                                   uint tid [[thread_position_in_threadgroup]],
                                   uint lane [[thread_index_in_simdgroup]],
                                   uint sg [[simdgroup_index_in_threadgroup]],
                                   uint row [[threadgroup_position_in_grid]]);
