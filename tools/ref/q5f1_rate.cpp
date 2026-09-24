// MUL-34 rate reference: ik's OWN Q5_1 kernel over the same synthetic shape
// qdot-rate benches (360,448 rows of K=10944 Q5_1 — the blk.0.ffn_down
// shape, one quantized column, single thread, 6 passes) — the number the
// Rust kernel must be judged against, measured on this box rather than
// quoted. Same kernel-table entry as q5f1_ref.cpp
// (mul_mat_qX_1_q8_2_T<Q5_1_Unpacker, 1>). K = 10944 keeps the TAIL path
// (2 blocks past the last x4 group) inside the timed loop — the shape the
// engine actually runs.
#include "ggml.h"
#include "ggml-backend.h"

#define IQK_IMPLEMENT
#include "iqk_gemm_legacy_quants.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

static const int kK = 10944; // dense ff width: 342 x 32-value blocks, 85 x4 groups + 2 tail
static const int kRows = 360448;
static const int kPasses = 6;

int main() {
    const size_t rs = 24 * (kK / 32); // 8208

    // Same bytes as qdot-rate's xorshift64* filler: any bytes are valid Q5_1 codes,
    // and the kernel's time is data-independent.
    unsigned long long s = 0x9E3779B97F4A7C15ull;
    auto next = [&]() {
        s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
        return s * 0x2545F4914F6CDD1Dull;
    };
    std::vector<uint8_t> w(kRows * rs);
    for (size_t i = 0; i + 8 <= w.size(); i += 8) { const unsigned long long v = next(); memcpy(&w[i], &v, 8); }
    std::vector<float> col(kK);
    for (int i = 0; i < kK; ++i) col[i] = (((long long)i % 31) - 15.0f) / 16.0f;
    const size_t ysz = 144 * (kK / 128) + 36 * ((kK % 128) / 32);
    std::vector<uint8_t> y(ysz);
    quantize_row_q8_2_x4(col.data(), y.data(), kK);

    std::array<mul_mat_t, IQK_MAX_NY> kernels{};
    mul_mat_t func16 = nullptr;
    if (!iqk_set_kernels_legacy_quants(kK, GGML_TYPE_Q5_1, GGML_TYPE_Q8_2_X4, kernels, func16) || !kernels[0]) {
        fprintf(stderr, "ik declined the Q5_1 x Q8_2_X4 pairing\n");
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
    printf("ik Q5_1 x4 rows %d x %zu B x %d passes in %.3f s = %.1f GB/s (dst[0] %g)\n",
           kRows, rs, kPasses, dt, bytes / dt / 1e9, dst[0]);
    return 0;
}
