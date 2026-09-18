// q3k_ref.cpp — reference harness for mulle stage 0.
//
// Opens the DeepSeek-V2-Lite-Chat Q3_K_M GGUF, extracts
// blk.1.ffn_gate_exps.weight, writes raw bytes + f32 activations to
// /root/mulle-data-muse/, computes CPU reference outputs by dequantizing
// with ggml_internal_get_type_traits(GGML_TYPE_Q3_K)->to_float and plain f32
// dots, then times ggml's CUDA mul_mat on four shapes and compares.
//
// Shapes (K=2048 fixed):
//   expert0 : one expert,  N=1408 rows,  1,239,040 weight bytes
//   stack   : all experts, N=90112 rows, 79,298,560 weight bytes
//   M = 1 and 8 activation columns each.
//
// Build: bash tools/ref/build.sh   (on the box, IK=/home/user/ik_llama.cpp)
// Run:   LD_LIBRARY_PATH=$IK/build/ggml/src ./tools/ref/q3k_ref

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

static const char * kTensorName = "blk.1.ffn_gate_exps.weight";
static const char * kGgufPath = "/models/small/DeepSeek-V2-Lite-Chat.Q3_K_M.gguf";
static const char * kDataDir = "/root/mulle-data-muse";

static void fail(const std::string & msg) {
    std::fprintf(stderr, "q3k_ref FATAL: %s\n", msg.c_str());
    std::exit(1);
}

static void write_file(const std::string & path, const void * data, size_t n) {
    FILE * f = std::fopen(path.c_str(), "wb");
    if (!f) fail("cannot open for write: " + path);
    if (std::fwrite(data, 1, n, f) != n) fail("short write: " + path);
    std::fclose(f);
}

static std::vector<uint8_t> read_range(const char * path, int64_t off, size_t n) {
    FILE * f = std::fopen(path, "rb");
    if (!f) fail(std::string("cannot open: ") + path);
    if (::fseeko(f, off, SEEK_SET) != 0) fail("seek failed");
    std::vector<uint8_t> out(n);
    if (std::fread(out.data(), 1, n, f) != n) fail("short read");
    std::fclose(f);
    return out;
}

struct Shape {
    const char * name;
    int64_t n; // rows
};

int main() {
    // ---- open GGUF, locate tensor metadata (no_alloc: metadata only) ----
    struct ggml_context * gctx = nullptr;
    struct gguf_init_params gp = {true, &gctx};
    struct gguf_context * gguf = gguf_init_from_file(kGgufPath, gp);
    if (!gguf || !gctx) fail("gguf_init_from_file failed");
    int tidx = gguf_find_tensor(gguf, kTensorName);
    if (tidx < 0) fail("tensor not found");
    struct ggml_tensor * info = ggml_get_tensor(gctx, kTensorName);
    if (!info) fail("ggml_get_tensor failed");
    if (info->type != GGML_TYPE_Q3_K) fail("tensor is not Q3_K");
    const int64_t K = info->ne[0];
    const int64_t N1 = info->ne[1];
    const int64_t NE = info->ne[2];
    std::printf("tensor %s type=Q3_K ne=[%lld,%lld,%lld]\n",
                kTensorName, (long long)K, (long long)N1, (long long)NE);
    if (K != 2048 || N1 != 1408 || NE != 64) fail("unexpected tensor dims");
    const size_t row_bytes = (size_t)(K / 256) * 110; // 8*110 = 880
    const size_t expert_bytes = (size_t)N1 * row_bytes; // 1,239,040
    const size_t total_bytes = expert_bytes * (size_t)NE; // 79,298,560
    if (ggml_nbytes(info) != total_bytes) fail("nbytes mismatch");

    // ---- read raw weight bytes through the GGUF API offsets ----
    const size_t data_off = gguf_get_data_offset(gguf);
    const size_t tensor_off = gguf_get_tensor_offset(gguf, tidx);
    std::vector<uint8_t> wAll =
        read_range(kGgufPath, (int64_t)(data_off + tensor_off), total_bytes);
    gguf_free(gguf);
    ggml_free(gctx);

    // ---- deterministic activations: mt19937 seed 1, uniform [-1,1] ----
    std::mt19937 rng(1);
    std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
    std::vector<float> x_m1((size_t)K);
    for (auto & v : x_m1) v = uni(rng);
    std::vector<float> x_m8((size_t)K * 8);
    for (auto & v : x_m8) v = uni(rng); // column c at offset c*K (column-major)

    // ---- CPU reference via ggml dequant traits + f32 dots ----
    ggml_type_traits_t traits = ggml_internal_get_type_traits(GGML_TYPE_Q3_K);
    std::vector<float> wrow((size_t)K);
    auto cpu_gemv = [&](const uint8_t * w, int64_t nrows, const float * x,
                        int m, std::vector<float> & y /* [nrows*m] row-major */) {
        y.assign((size_t)nrows * m, 0.0f);
        for (int64_t r = 0; r < nrows; r++) {
            traits.to_float(w + (size_t)r * row_bytes, wrow.data(), K);
            for (int c = 0; c < m; c++) {
                const float * xc = x + (size_t)c * (size_t)K;
                double acc = 0.0;
                for (int64_t k = 0; k < K; k++) acc += (double)wrow[(size_t)k] * xc[k];
                y[(size_t)r * m + c] = (float)acc;
            }
        }
    };
    std::vector<float> y_e0_m1, y_e0_m8, y_st_m1, y_st_m8;
    cpu_gemv(wAll.data(), N1, x_m1.data(), 1, y_e0_m1);
    cpu_gemv(wAll.data(), N1, x_m8.data(), 8, y_e0_m8);
    cpu_gemv(wAll.data(), N1 * NE, x_m1.data(), 1, y_st_m1);
    cpu_gemv(wAll.data(), N1 * NE, x_m8.data(), 8, y_st_m8);

    // ---- write data files ----
    write_file(std::string(kDataDir) + "/gate.q3k", wAll.data(), total_bytes);
    write_file(std::string(kDataDir) + "/x_m1.f32", x_m1.data(),
               x_m1.size() * sizeof(float));
    write_file(std::string(kDataDir) + "/x_m8.f32", x_m8.data(),
               x_m8.size() * sizeof(float));
    write_file(std::string(kDataDir) + "/y_ref_expert0_m1.f32", y_e0_m1.data(),
               y_e0_m1.size() * sizeof(float));
    write_file(std::string(kDataDir) + "/y_ref_expert0_m8.f32", y_e0_m8.data(),
               y_e0_m8.size() * sizeof(float));
    write_file(std::string(kDataDir) + "/y_ref_stack_m1.f32", y_st_m1.data(),
               y_st_m1.size() * sizeof(float));
    write_file(std::string(kDataDir) + "/y_ref_stack_m8.f32", y_st_m8.data(),
               y_st_m8.size() * sizeof(float));
    std::printf("wrote data files to %s\n", kDataDir);

    auto max_abs = [](const std::vector<float> & v) {
        double m = 0;
        for (float f : v) m = std::max(m, (double)std::fabs(f));
        return m;
    };

    // ---- CUDA backend timing ----
    ggml_backend_t backend = ggml_backend_cuda_init(0, nullptr, nullptr);
    if (!backend) fail("ggml_backend_cuda_init failed");

    struct Case {
        const char * name;
        const uint8_t * w;
        int64_t nrows;
        const float * x;
        int m;
        const std::vector<float> & yref;
    };
    const Case cases[4] = {
        {"expert0_m1", wAll.data(), N1, x_m1.data(), 1, y_e0_m1},
        {"expert0_m8", wAll.data(), N1, x_m8.data(), 8, y_e0_m8},
        {"stack_m1", wAll.data(), N1 * NE, x_m1.data(), 1, y_st_m1},
        {"stack_m8", wAll.data(), N1 * NE, x_m8.data(), 8, y_st_m8},
    };
    for (const Case & cs : cases) {
        // no_alloc: tensor storage comes from ggml_backend_alloc_ctx_tensors
        struct ggml_init_params mp = {256u << 20, nullptr, true};
        struct ggml_context * ctx = ggml_init(mp);
        struct ggml_tensor * w =
            ggml_new_tensor_2d(ctx, GGML_TYPE_Q3_K, K, cs.nrows);
        struct ggml_tensor * x =
            ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, cs.m);
        struct ggml_tensor * y = ggml_mul_mat(ctx, w, x);
        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, y);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buf) fail("alloc_ctx_tensors failed");
        const size_t wn = (size_t)cs.nrows * row_bytes;
        ggml_backend_tensor_set(w, cs.w, 0, wn);
        ggml_backend_tensor_set(x, cs.x, 0, (size_t)K * cs.m * sizeof(float));

        for (int i = 0; i < 20; i++) ggml_backend_graph_compute(backend, gf);
        ggml_backend_synchronize(backend);
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < 200; i++) ggml_backend_graph_compute(backend, gf);
        ggml_backend_synchronize(backend);
        auto t1 = std::chrono::steady_clock::now();
        double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / 200.0;

        // ggml y layout is [N x M]: element (r,c) at c*N + r
        std::vector<float> yg((size_t)cs.nrows * cs.m);
        ggml_backend_tensor_get(y, yg.data(), 0, yg.size() * sizeof(float));
        double denom = max_abs(cs.yref);
        double maxerr = 0;
        for (int64_t r = 0; r < cs.nrows; r++)
            for (int c = 0; c < cs.m; c++) {
                double d = std::fabs((double)yg[(size_t)c * cs.nrows + r] -
                                     cs.yref[(size_t)r * cs.m + c]);
                maxerr = std::max(maxerr, d);
            }
        double gbs = (double)wn / (us * 1e-6) / 1e9;
        std::printf("shape %-10s N=%6lld M=%d weight_bytes=%9zu us=%9.2f GB/s=%7.2f max_rel_err=%.3e\n",
                    cs.name, (long long)cs.nrows, cs.m, wn, us, gbs, maxerr / denom);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
    ggml_backend_free(backend);
    return 0;
}
