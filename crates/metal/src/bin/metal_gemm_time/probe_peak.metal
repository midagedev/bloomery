// Probe-only: the machine's own ruler for metal_gemm_time (`op=peak_*`).
// Kernels with no memory traffic in their loops, so a rate printed for one
// is the rate of the pipe it loads, at the clock the probe's process gets:
//   probe_peak_mma_f32 eight f32 8x8 simdgroup multiply-accumulates in flight
//   probe_peak_mma_f16 the same, half operands into the float accumulator
// This text is compiled into the reference arm's library; it never enters
// the engine's kernel source.

#include <metal_stdlib>
#include <metal_simdgroup_matrix>

using namespace metal;


#define PEAK_MMA_ITERS 1024

// The operands load from memory each round, at an address that depends on
// the round: the product of two matrices the loop does not change would be
// hoisted out of it. Two loads feed eight multiply-accumulates.
template <typename T>
kernel void probe_peak_mma(const device T * in [[buffer(0)]],
                           device float * out [[buffer(1)]],
                           uint tgid [[threadgroup_position_in_grid]],
                           uint sgid [[simdgroup_index_in_threadgroup]]) {
    simdgroup_matrix<float, 8, 8> c[8];
    for (short i = 0; i < 8; ++i) {
        c[i] = simdgroup_matrix<float, 8, 8>(0.f);
    }
    for (int it = 0; it < PEAK_MMA_ITERS; ++it) {
        simdgroup_matrix<T, 8, 8> a;
        simdgroup_matrix<T, 8, 8> b;
        simdgroup_load(a, in + (it & 1) * 64, 8);
        simdgroup_load(b, in + 128 + (it & 1) * 64, 8);
        _Pragma("clang loop unroll(full)")
        for (short i = 0; i < 8; ++i) {
            simdgroup_multiply_accumulate(c[i], a, b, c[i]);
        }
    }
    for (short i = 0; i < 8; ++i) {
        simdgroup_store(c[i], out + ((tgid * 4 + sgid) * 8 + i) * 64, 8);
    }
}

template [[host_name("probe_peak_mma_f32")]] [[kernel]] decltype(probe_peak_mma<float>) probe_peak_mma<float>;
template [[host_name("probe_peak_mma_f16")]] [[kernel]] decltype(probe_peak_mma<half>) probe_peak_mma<half>;
