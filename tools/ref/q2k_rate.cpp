// Rate reference: ik's OWN Q2_K kernel over the same synthetic shape qdot-rate benches for
// Q2_K (360,448 rows of K=4096 — the IQ3_S row's k, so the per-byte rates compare — one
// quantized column, single thread, 6 passes): the number the Rust kernel must beat, measured on
// this box rather than quoted. Same kernel-table entry as q2k_ref.cpp
// (mul_mat_qY_K_q8_K_T<DequantizerQ2K, 1> through iqk_set_kernels_kquants). Every block's two
// f16 fields (d and dmin) are masked finite, as qdot-rate does.
#include "ggml.h"
#include "ggml-backend.h"

#define IQK_IMPLEMENT
#include "iqk_gemm_kquants.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

static const int kK = 4096;
static const int kRows = 360448;
static const int kPasses = 6;

int main() {
    const size_t rs = 84 * (kK / 256);

    // Same bytes as qdot-rate's xorshift64* filler: any bytes are valid Q2_K codes (scales, mins
    // and 2-bit codes are any bits), and the kernel's time is data-independent; then every
    // block's f16 d and dmin (the high bytes, at 81 and 83) masked finite: raw bytes give an
    // infinity or NaN in one block of 16 on average.
    unsigned long long s = 0x9E3779B97F4A7C15ull;
    auto next = [&]() {
        s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
        return s * 0x2545F4914F6CDD1Dull;
    };
    std::vector<uint8_t> w((size_t)kRows * rs);
    for (size_t i = 0; i + 8 <= w.size(); i += 8) { const unsigned long long v = next(); memcpy(&w[i], &v, 8); }
    for (size_t b = 0; b < w.size(); b += 84) { w[b + 81] &= 0x7b; w[b + 83] &= 0x7b; }
    std::vector<float> col(kK);
    for (int i = 0; i < kK; ++i) col[i] = (((long long)i % 31) - 15.0f) / 16.0f;
    ggml_type_traits_t q8k = ggml_internal_get_type_traits(GGML_TYPE_Q8_K);
    std::vector<uint8_t> y(ggml_row_size(GGML_TYPE_Q8_K, kK));
    q8k.from_float(col.data(), y.data(), kK);

    std::array<mul_mat_t, IQK_MAX_NY> kernels{};
    mul_mat_t func16 = nullptr;
    if (!iqk_set_kernels_kquants(kK, GGML_TYPE_Q2_K, GGML_TYPE_Q8_K, kernels, func16) || !kernels[0]) {
        fprintf(stderr, "ik declined the Q2_K x Q8_K pairing\n");
        return 1;
    }
    std::vector<float> dst(kRows);
    DataInfo info;
    info.s = dst.data();
    info.cy = (const char *)y.data();
    info.bs = kRows;
    info.by = y.size();
    info.cur_y = 0;
    info.ne11 = 1;
    info.row_mapping = nullptr;
    info.bs2 = 0;

    // Warm-up, then timed passes; dst feeds the report so nothing is dead.
    kernels[0](kK, w.data(), rs, info, 4096);
    auto t0 = std::chrono::steady_clock::now();
    for (int p = 0; p < kPasses; ++p) kernels[0](kK, w.data(), rs, info, kRows);
    double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    double bytes = (double)kRows * rs * kPasses;
    printf("ik Q2_K rows %d x %zu B x %d passes in %.3f s = %.1f GB/s (dst[0] %g)\n",
           kRows, rs, kPasses, dt, bytes / dt / 1e9, dst[0]);
    return 0;
}
