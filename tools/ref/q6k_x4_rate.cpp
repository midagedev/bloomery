// MUL-31 rate reference: ik's OWN Q6_K x4 kernel over the same synthetic
// shape qdot-rate benches (360,448 rows of K=2048 Q6_K, one quantized
// column, single thread, 6 passes) — the number the Rust kernel is judged
// against, measured on this box rather than quoted. Same kernel-table
// entry as q6k_x4_ref.cpp (mul_mat_qY_K_q8_2_X4_T<DequantizerQ6K_AVX2, 1>).
#include "ggml.h"
#include "ggml-backend.h"

#define IQK_IMPLEMENT
#include "iqk_gemm_kquants.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

static const int kK = 2048;
static const int kRows = 360448;
static const int kPasses = 6;

int main() {
    const size_t rs = 210 * (kK / 256);

    // Same xorshift64* filler as qdot-rate: any bytes are valid Q6_K codes,
    // and the kernel's time is data-independent.
    unsigned long long s = 0x9E3779B97F4A7C15ull;
    auto next = [&]() {
        s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
        return s * 0x2545F4914F6CDD1Dull;
    };
    std::vector<uint8_t> w(kRows * rs);
    for (size_t i = 0; i + 8 <= w.size(); i += 8) memcpy(&w[i], &s, 8), next();
    std::vector<float> col(kK);
    for (int i = 0; i < kK; ++i) col[i] = (((long long)i % 31) - 15.0f) / 16.0f;
    std::vector<uint8_t> y(144 * (kK / 128));
    quantize_row_q8_2_x4(col.data(), y.data(), kK);

    std::array<mul_mat_t, IQK_MAX_NY> kernels{};
    mul_mat_t func16 = nullptr;
    if (!iqk_set_kernels_kquants(kK, GGML_TYPE_Q6_K, GGML_TYPE_Q8_2_X4, kernels, func16) || !kernels[0]) {
        fprintf(stderr, "ik declined the Q6_K x Q8_2_X4 pairing\n");
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
    printf("ik Q6_K x4 rows %d x %zu B x %d passes in %.3f s = %.1f GB/s (dst[0] %g)\n",
           kRows, rs, kPasses, dt, bytes / dt / 1e9, dst[0]);
    return 0;
}
