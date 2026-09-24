// Rate reference: ik's OWN IQ3_XXS kernel over the same synthetic shape qdot-rate benches
// for IQ3_XXS (360,448 rows of K=4096 — the V4-Flash ffn_gate/up_exps row length — one
// quantized column, single thread, 6 passes): the number the Rust kernel must beat,
// measured on this box rather than quoted. Same kernel-table entry as iq3xxs_ref.cpp
// (mul_mat_qX_K_q8_K_IQ_N<DequantizerIQ3XXS, 1> through iqk_set_kernels_iquants).
#include "ggml.h"
#include "ggml-backend.h"

#define IQK_IMPLEMENT
#include "iqk_gemm_iquants.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

static const int kK = 4096;
static const int kRows = 360448;
static const int kPasses = 6;

int main() {
    const size_t rs = 98 * (kK / 256);

    // Same xorshift64* filler as qdot-rate: any bytes are valid IQ3_XXS codes (grid
    // indices are bytes, signs and scales are any word bits), and the kernel's time is
    // data-independent.
    unsigned long long s = 0x9E3779B97F4A7C15ull;
    auto next = [&]() {
        s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
        return s * 0x2545F4914F6CDD1Dull;
    };
    std::vector<uint8_t> w((size_t)kRows * rs);
    for (size_t i = 0; i + 8 <= w.size(); i += 8) memcpy(&w[i], &s, 8), next();
    std::vector<float> col(kK);
    for (int i = 0; i < kK; ++i) col[i] = (((long long)i % 31) - 15.0f) / 16.0f;
    ggml_type_traits_t q8k = ggml_internal_get_type_traits(GGML_TYPE_Q8_K);
    std::vector<uint8_t> y(ggml_row_size(GGML_TYPE_Q8_K, kK));
    q8k.from_float(col.data(), y.data(), kK);

    std::array<mul_mat_t, IQK_MAX_NY> kernels{};
    mul_mat_t func16 = nullptr;
    if (!iqk_set_kernels_iquants(kK, GGML_TYPE_IQ3_XXS, GGML_TYPE_Q8_K, kernels, func16) || !kernels[0]) {
        fprintf(stderr, "ik declined the IQ3_XXS x Q8_K pairing\n");
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
    printf("ik IQ3_XXS rows %d x %zu B x %d passes in %.3f s = %.1f GB/s (dst[0] %g)\n",
           kRows, rs, kPasses, dt, bytes / dt / 1e9, dst[0]);
    return 0;
}
