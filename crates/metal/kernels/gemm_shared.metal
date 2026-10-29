// What the two GEMM libraries share: the engine library's simdgroup GEMMs
// (gemm.metal, gemm_routed.metal) and the tensor-op GEMMs of the library
// compiled at Metal 4 (gemm_tensor.metal). The tile's geometry and the
// argument blocks the host packs are one program's ABI whichever library
// runs it (kernels/gemm.rs and kernels/routed.rs launch both), and the
// expert pair's SwiGLU is the decode pair's (experts.metal) and both
// grouped GEMMs' epilogue. The activations' checked cast is every GEMM's.

#include <metal_stdlib>

using namespace metal;

#pragma METAL fp contract(off)

// The tile's geometry: the kernel's own constants, the ones the host's
// launch module (kernels/gemm.rs) mirrors.
#define GEMM_ROWS 64     // weight rows a threadgroup owns
#define GEMM_ACTS 64     // activation rows a threadgroup owns
#define GEMM_THREADS 128  // four SIMD groups

// The K values one step covers: 32, a whole number of every owner's
// 16-value staging chunks and half the smallest group (A4's 64).
template <typename T>
constexpr int gemm_bk() {
    return 32;
}

// The staged rows' pitch in elements: the K step plus 16 bytes, steel's
// padding.
template <typename T>
constexpr int gemm_lda() {
    return gemm_bk<T>() + 16 / (int)sizeof(T);
}

// The threadgroup bytes one tile stages: the f32 arm's 18432, the f16
// arm's 10240. The allocation itself lives in the entry kernel (a
// threadgroup variable may be declared only there), and the tile body takes
// it as a pointer.
template <typename T>
constexpr int gemm_shmem_bytes() {
    return (GEMM_ROWS + GEMM_ACTS) * gemm_lda<T>() * (int)sizeof(T);
}

// The argument block of a plain GEMM, bound by value at slot 3.
struct GemmArgs {
    uint cols;     // the input columns K, a whole number of the type's blocks
    uint rows;     // the weight rows N, the output's columns
    uint acts;     // the activation rows R, the output's rows
    uint x_stride; // the elements between x's rows (f32, or half for an
                   // _h16 entry), at least cols
    uint y_stride; // the f32 elements between y's rows, at least rows
    uint layer;    // the layer a raise is recorded in
};

static_assert(sizeof(GemmArgs) == 24, "GemmArgs is 24 bytes: the host packs six words");

// One segment of a segmented GEMM: the weight rows of one tensor, and where
// they sit in the weights and in the output. The mat-vec's segment table's
// own four fields (norm.metal's MatvecSeg), the GEMM's meaning: the output
// offset is a column offset in the R-row output, and the first group is a
// weight-tile threadgroup.
struct GemmSeg {
    uint rows;         // the segment's weight rows
    uint wblocks;      // the segment's first row's block offset in the weights
    uint out;          // the segment's first column in the output
    uint first_group;  // the segment's first weight-tile threadgroup
};

// The argument block of a segmented GEMM, bound by value at slot 3.
struct GemmSegArgs {
    uint cols;
    uint acts;
    uint x_stride;
    uint y_stride;
    uint layer;
    uint nseg;
    GemmSeg segs[BLOOMERY_MATVEC_MAX_SEGMENTS];
};

static_assert(sizeof(GemmSegArgs) == 24 + 16 * BLOOMERY_MATVEC_MAX_SEGMENTS,
              "GemmSegArgs is the header's six words and the segments' tables");

// The routed tile's geometry (gemm_routed.metal, gemm_tensor.metal): the
// host's launch module (kernels/routed.rs) mirrors it.
#define ROUTED_ROWS 64   // weight rows of a tile
#define ROUTED_ACTS 64   // activation rows of a tile

// Eight words, bound by value.
struct RoutedArgs {
    uint cols;          // the input columns K
    uint x_stride;      // f32 elements between the activation rows
    uint y_stride;      // f32 elements between the output rows
    uint layer;
    uint n_experts;     // the stack's experts; the table's group n_experts is shared
    uint stack_blocks;  // blocks of one expert's rows in the stack
    uint gather;        // 1: an activation row is the slot's pass row; 0: the place's
    uint rows;          // outputs per expert: the pair's rows of gate, the down's rows
};

static_assert(sizeof(RoutedArgs) == 32, "RoutedArgs is eight words");

// silu(gate) * up. No route weight: the fold multiplies by it.
inline float pair_swiglu(float gate, float up) {
    return gate / (1.0f + exp(-gate)) * up;
}

// The f16 operand's cast of four activations, the activation quantizer's: a
// value that is not finite or past the half range (one test: not at most the
// half maximum in magnitude, which a NaN fails too) raises the quantizer's
// site and casts as NaN, so the row's outputs are NaN — never a silent inf.
// Every GEMM of both libraries casts its activations through this one cast,
// in its staging or in the half-input GEMMs' own pass (gemm_cast_rows).
inline half4 gemm_half4(float4 v, device atomic_uint * faults, uint layer) {
    const bool4 bad = !(fabs(v) <= 65504.f);
    if (any(bad)) {
        raise(faults, layer, BLOOMERY_FAULT_SITE_QUANT_COLUMN);
        return half4(select(v, float4(NAN), bad));
    }
    return half4(v);
}

// One 16-value run of an activation row, staged for a GEMM's tile: loaded
// as scalars (no window alignment is assumed) and composed into float4s, the
// f16 arm's through the checked cast (gemm_half4).
template <typename T>
inline void gemm_stage_x(device const float * xr, threadgroup vec<T, 4> * dst,
                         device atomic_uint * faults, uint layer) {
    FOR_UNROLL(short g = 0; g < 4; ++g) {
        const float4 v = float4(xr[4 * g], xr[4 * g + 1], xr[4 * g + 2], xr[4 * g + 3]);
        if (sizeof(T) == 2) {
            dst[g] = vec<T, 4>(gemm_half4(v, faults, layer));
        } else {
            dst[g] = vec<T, 4>(v);
        }
    }
}

// One 16-value run of a half activation row, the checked cast already run
// over it (gemm_cast_rows): four half4 loads, staged as they are. The rows
// are 8-byte aligned and their stride a multiple of four (kernels/gemm.rs
// refuses any other window). Only the f16 arm stages half rows.
template <typename T>
inline void gemm_stage_x(device const half * xr, threadgroup vec<T, 4> * dst,
                         device atomic_uint *, uint) {
    static_assert(sizeof(T) == 2, "half rows stage into the f16 arm only");
    FOR_UNROLL(short g = 0; g < 4; ++g) {
        dst[g] = *(device const half4 *)(xr + 4 * g);
    }
}
