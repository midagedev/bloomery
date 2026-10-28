// The prompt pass's routing and the row-wise glue around the experts. One
// source, the entries under their host names:
//
//   route_rows_select  rows of router logits -> each row's ids and weights:
//                      a threadgroup a row, the decode router's selection
//                      (route_pick, router.metal) over the row's logits
//   route_table        the counting sort of every slot of the pass by expert:
//                      the slot list in expert order, each slot's place in
//                      it, and the table of tiles the grouped GEMM walks
//   combine_rows       x[row] += y[slot 0] + ... + y[slot per_row - 1], the
//                      slots' outputs read through each slot's place in the
//                      sorted list, summed in slot order
//   headnorm_rows      the gated delta numerator times each head's norm
//                      scale, in place: the head-norm mat-vec's input
//   swiglu_rows        a dense FFN's gate rows replaced by silu(gate) * up,
//                      the experts' SwiGLU (pair_swiglu) over two dense GEMMs'
//                      outputs
//
// The route table. A pass of R rows has 9 R slots: a row's eight picks and
// the shared expert, whose id is the router's 256 (an instance with no gated
// shared expert has 8 R slots and no shared group). One threadgroup counts
// the slots per group (the 256 routed experts, then the shared expert as the
// 257th group), scans the counts, and places each slot at its group's start
// plus the slot's arrival at the group (a threadgroup atomic). The order
// inside a group is therefore not fixed; nothing reads it: a slot's output is
// written at the slot's own place and read through `inv`, and a tile's rows
// are independent dot products. The tiles cut each group into runs of at most
// ROUTE_TILE slots: the table's first word is the tile count, then one four-
// word entry (group, first place, slots, 0) each. The launch grid is the
// table's capacity, ceil(9 R / ROUTE_TILE) + 257 entries, and a threadgroup
// past the count returns.
//
// An id outside its slot's range (a routed slot's past the stack, the shared
// slot's not the shared id) raises the expert-id site and is placed in group 0,
// a defined access; the pass is poisoned by the raise.
//
// This file comes after router.metal (the selection and its geometry),
// experts.metal's neighbours in the build order, and fault.metal.

#include <metal_stdlib>

using namespace metal;

#pragma METAL fp contract(off)

// Slots a tile of the grouped GEMM holds at most: the GEMM's activation tile.
#define ROUTE_TILE 64
// Threads of the table's one threadgroup.
#define ROUTE_THREADS 1024
// Groups: the routed experts and a gated shared expert.
#define ROUTE_GROUPS (ROUTER_EXPERTS + BLOOMERY_ROUTER_SHARED)

struct RowsPickArgs {
    uint top_k;
    uint layer;
    uint stride;  // floats between rows of the logits
};

static_assert(sizeof(RowsPickArgs) == 12, "RowsPickArgs is three words");

// The selection of one row of logits. The row's logits are published to
// threadgroup memory with their packed keys and "bad" flags as the decode's
// last threadgroup publishes them, then the first SIMD group selects.
kernel void route_rows_select(device const float *logits [[buffer(0)]],
                              device uint *ids [[buffer(1)]],
                              device float *weights [[buffer(2)]],
                              constant RowsPickArgs &args [[buffer(3)]],
                              device atomic_uint *faults [[buffer(4)]],
                              uint row [[threadgroup_position_in_grid]],
                              uint tid [[thread_position_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]],
                              uint sg_in_tg [[simdgroup_index_in_threadgroup]])
{
    threadgroup float tg_logits[ROUTER_ROWS];
    threadgroup ulong tg_keys[ROUTER_EXPERTS];
    threadgroup uint tg_bad[ROUTER_SG];

    device const float *lr = logits + (ulong)row * args.stride;
    uint mine_bad = 0u;
    for (uint r = tid; r < ROUTER_ROWS; r += ROUTER_THREADS) {
        const float l = lr[r];
        tg_logits[r] = l;
        mine_bad |= uint(!isfinite(l));
        if (r < ROUTER_EXPERTS) {
            tg_keys[r] = argmax_key(l, r);
        }
    }
    mine_bad = simd_max(mine_bad);
    if (lane == 0u) {
        tg_bad[sg_in_tg] = mine_bad;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_in_tg != 0u) {
        return;
    }
    const uint per_row = args.top_k + BLOOMERY_ROUTER_SHARED;
    route_pick(ids + (ulong)row * per_row, weights + (ulong)row * per_row, nullptr, faults,
               args.top_k, args.layer, lane, tg_logits, tg_keys, tg_bad);
}

struct RouteTableArgs {
    uint slots;      // the pass's slots: 9 rows
    uint per_row;    // slots a row holds
    uint n_experts;  // the routed stack's experts; the shared group follows
    uint max_tiles;  // the table's capacity in entries
    uint layer;
};

static_assert(sizeof(RouteTableArgs) == 20, "RouteTableArgs is five words");

// The table's one threadgroup. `ids` is a slot's group, `sorted` the slot at
// a place, `inv` the place of a slot, `tiles` the table.
kernel void route_table(device const uint *ids [[buffer(0)]],
                        device uint *sorted [[buffer(1)]],
                        device uint *inv [[buffer(2)]],
                        device uint4 *tiles [[buffer(3)]],
                        constant RouteTableArgs &args [[buffer(4)]],
                        device atomic_uint *faults [[buffer(5)]],
                        uint tid [[thread_position_in_threadgroup]])
{
    threadgroup atomic_uint count[ROUTE_GROUPS];
    threadgroup uint start[ROUTE_GROUPS];
    threadgroup uint first_tile[ROUTE_GROUPS];
    threadgroup uint n_tiles;

    for (uint g = tid; g < ROUTE_GROUPS; g += ROUTE_THREADS) {
        atomic_store_explicit(&count[g], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // The group of a slot: its id, or group 0 for an id outside the slot's
    // range, raised.
    for (uint s = tid; s < args.slots; s += ROUTE_THREADS) {
        const uint id = ids[s];
        const bool shared = BLOOMERY_ROUTER_SHARED && (s % args.per_row) == args.per_row - 1u;
        const bool ok = shared ? id == args.n_experts : id < args.n_experts;
        if (!ok) {
            raise(faults, args.layer, BLOOMERY_FAULT_SITE_EXPERT_ID);
        }
        atomic_fetch_add_explicit(&count[ok ? id : 0u], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // The scan, serial: 257 groups.
    if (tid == 0u) {
        uint place = 0u;
        uint tile = 0u;
        for (uint g = 0u; g < ROUTE_GROUPS; ++g) {
            const uint n = atomic_load_explicit(&count[g], memory_order_relaxed);
            start[g] = place;
            first_tile[g] = tile;
            place += n;
            tile += (n + ROUTE_TILE - 1u) / ROUTE_TILE;
        }
        n_tiles = tile;
        tiles[0] = uint4(tile, 0u, 0u, 0u);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // The table's entries: a group's thread writes the group's tiles.
    for (uint g = tid; g < ROUTE_GROUPS; g += ROUTE_THREADS) {
        const uint n = atomic_load_explicit(&count[g], memory_order_relaxed);
        for (uint j = 0u; j * ROUTE_TILE < n; ++j) {
            tiles[1u + first_tile[g] + j] =
                uint4(g, start[g] + j * ROUTE_TILE, min((uint)ROUTE_TILE, n - j * ROUTE_TILE), 0u);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // The place of each slot: the group's start plus the arrival.
    for (uint g = tid; g < ROUTE_GROUPS; g += ROUTE_THREADS) {
        atomic_store_explicit(&count[g], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tid; s < args.slots; s += ROUTE_THREADS) {
        const uint id = ids[s];
        const bool shared = BLOOMERY_ROUTER_SHARED && (s % args.per_row) == args.per_row - 1u;
        const bool ok = shared ? id == args.n_experts : id < args.n_experts;
        const uint g = ok ? id : 0u;
        const uint place =
            start[g] + atomic_fetch_add_explicit(&count[g], 1u, memory_order_relaxed);
        sorted[place] = s;
        inv[s] = place;
    }
}

struct CombineArgs {
    uint cols;
    uint per_row;
    uint x_stride;
    uint y_stride;
};

static_assert(sizeof(CombineArgs) == 16, "CombineArgs is four words");

// x[row] += y[place of slot 0] + y[place of slot 1] + ... in slot order,
// the sum started at zero and added to the residual once. A threadgroup a row.
kernel void combine_rows(device const float *y [[buffer(0)]],
                         device const uint *inv [[buffer(1)]],
                         device float *x [[buffer(2)]],
                         constant CombineArgs &args [[buffer(3)]],
                         uint row [[threadgroup_position_in_grid]],
                         uint tid [[thread_position_in_threadgroup]],
                         uint threads [[threads_per_threadgroup]])
{
    device float4 *xr = (device float4 *)(x + (ulong)row * args.x_stride);
    for (uint f4 = tid; f4 < args.cols / 4u; f4 += threads) {
        float4 acc = float4(0.f);
        for (uint s = 0u; s < args.per_row; ++s) {
            const ulong place = inv[(ulong)row * args.per_row + s];
            acc += *((device const float4 *)(y + place * args.y_stride) + f4);
        }
        xr[f4] = xr[f4] + acc;
    }
}

struct HeadNormRowsArgs {
    uint g_stride;
    uint partials_stride;
    float eps;
    uint layer;
};

static_assert(sizeof(HeadNormRowsArgs) == 16, "HeadNormRowsArgs is four words");

// g[row][head][col] *= 1 / sqrt(sum of the head's partials / HEAD_COLS + eps),
// the head-norm mat-vec's scale and its refusal: a head the norm cannot take
// raises the linear-gate site and is NaN. A threadgroup a (row, head), a
// thread a column.
kernel void headnorm_rows(device float *g [[buffer(0)]],
                          device const float *partials [[buffer(1)]],
                          constant HeadNormRowsArgs &args [[buffer(2)]],
                          device atomic_uint *faults [[buffer(3)]],
                          uint2 tg [[threadgroup_position_in_grid]],
                          uint2 tpg [[thread_position_in_threadgroup]])
{
    const uint row = tg.x;
    const uint head = tg.y;
    const uint tid = tpg.x;
    device const float *p = partials + (ulong)row * args.partials_stride +
                            head * BLOOMERY_DELTA_COLUMN_BLOCKS;
    float sum = 0.f;
    for (short j = 0; j < BLOOMERY_DELTA_COLUMN_BLOCKS; ++j) {
        sum += p[j];
    }
    const float cols = float(BLOOMERY_DELTA_HEAD_COLS);
    const float mean = sum / cols;
    float s = rsqrt(mean + args.eps);
    if (!isfinite(sum) || !isfinite(s)) {
        if (tid == 0u) {
            const uint site = BLOOMERY_FAULT_SITE_LINEAR_GATE;
            raise(faults, args.layer, site);
        }
        s = NAN;
    }
    g[(ulong)row * args.g_stride + head * BLOOMERY_DELTA_HEAD_COLS + tid] *= s;
}

struct SwigluRowsArgs {
    uint cols;
    uint stride;
};

static_assert(sizeof(SwigluRowsArgs) == 8, "SwigluRowsArgs is two words");

// gate[row][c] = silu(gate[row][c]) * up[row][c], in place over the gate's
// rows: the pair's epilogue (pair_swiglu, the decode pair's and the grouped
// pair tile's) over the two outputs of a dense FFN's gate and up GEMMs. A
// threadgroup is a run of columns of one row, a thread a column.
kernel void swiglu_rows(device float *gate [[buffer(0)]],
                        device const float *up [[buffer(1)]],
                        constant SwigluRowsArgs &args [[buffer(2)]],
                        uint2 tg [[threadgroup_position_in_grid]],
                        uint2 tpg [[thread_position_in_threadgroup]],
                        uint2 ntg [[threads_per_threadgroup]])
{
    const uint col = tg.x * ntg.x + tpg.x;
    if (col >= args.cols) {
        return;
    }
    const ulong at = (ulong)tg.y * (ulong)args.stride + (ulong)col;
    gate[at] = pair_swiglu(gate[at], up[at]);
}
