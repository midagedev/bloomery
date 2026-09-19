// q3k_cpu_ref.cpp — CPU reference + timing harness for the mulle stage-2 pre-study.
//
// Opens the DeepSeek-V2-Lite-Chat Q3_K_M GGUF through the GGUF API (never
// hand-computed offsets), extracts blk.1.ffn_gate_exps (the `expert0`/`stack`
// shapes, byte-identical to what tools/ref/q3k_ref.cpp writes to gate.q3k)
// plus the three sibling tensors forming the `big` shape, generates the same
// deterministic activations as q3k_ref.cpp (mt19937 seed 1, x_m1 first, then
// x_m8), computes f32 references on the CPU via
// ggml_internal_get_type_traits(GGML_TYPE_Q3_K)->to_float + f64 dots, writes
// everything to $MULLE_DATA, then times ik_llama.cpp's CPU backend
// (ggml_backend_cpu_init + ggml_backend_cpu_set_n_threads, one ggml_mul_mat
// graph, Q3_K weights x F32 activations) over shapes x M x thread counts.
//
// The data-file bytes are deterministic, so this binary is self-sufficient:
// it reproduces gate.q3k/x_m1.f32/x_m8.f32 bit-identically instead of running
// the CUDA harness (which would contend the 3090 another track measures on).
//
// Usage:
//   q3k_cpu_ref [--data-only]   # with no args: write data, then time
//
// Build: bash tools/ref/build-cpu.sh   (on the box, IK=/home/user/ik_llama.cpp)

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"

static const char * kGgufPath = "/models/small/DeepSeek-V2-Lite-Chat.Q3_K_M.gguf";
static const char * kDataDir = getenv("MULLE_DATA") ? getenv("MULLE_DATA") : "/root/mulle-data";

static const char * kGateName = "blk.1.ffn_gate_exps.weight";
static const char * kBigNames[4] = {
    "blk.1.ffn_gate_exps.weight",
    "blk.1.ffn_up_exps.weight",
    "blk.2.ffn_gate_exps.weight",
    "blk.2.ffn_up_exps.weight",
};

static void fail(const std::string & msg) {
    std::fprintf(stderr, "q3k_cpu_ref FATAL: %s\n", msg.c_str());
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

static double max_abs(const std::vector<float> & v) {
    double m = 0;
    for (float f : v) m = std::max(m, (double)std::fabs(f));
    return m;
}

int main(int argc, char ** argv) {
    const bool data_only = argc > 1 && std::string(argv[1]) == "--data-only";

    // ---- open GGUF, locate tensor metadata (no_alloc: metadata only) ----
    struct ggml_context * gctx = nullptr;
    struct gguf_init_params gp = {true, &gctx};
    struct gguf_context * gguf = gguf_init_from_file(kGgufPath, gp);
    if (!gguf || !gctx) fail("gguf_init_from_file failed");

    const size_t data_off = gguf_get_data_offset(gguf);
    std::vector<uint8_t> parts[4];
    int64_t K = 0;
    for (int p = 0; p < 4; p++) {
        int tidx = gguf_find_tensor(gguf, kBigNames[p]);
        if (tidx < 0) fail(std::string("tensor not found: ") + kBigNames[p]);
        struct ggml_tensor * info = ggml_get_tensor(gctx, kBigNames[p]);
        if (!info) fail("ggml_get_tensor failed");
        if (info->type != GGML_TYPE_Q3_K) fail(std::string("not Q3_K: ") + kBigNames[p]);
        if (p == 0) {
            K = info->ne[0];
            std::printf("tensor %s type=Q3_K ne=[%lld,%lld,%lld]\n",
                        kBigNames[p], (long long)info->ne[0],
                        (long long)info->ne[1], (long long)info->ne[2]);
            if (K != 2048 || info->ne[1] != 1408 || info->ne[2] != 64)
                fail("unexpected gate tensor dims");
        } else if (info->ne[0] != K || info->ne[1] != 1408 || info->ne[2] != 64) {
            fail(std::string("dim mismatch: ") + kBigNames[p]);
        }
        const size_t tensor_off = gguf_get_tensor_offset(gguf, tidx);
        const size_t nbytes = ggml_nbytes(info);
        parts[p] = read_range(kGgufPath, (int64_t)(data_off + tensor_off), nbytes);
        std::printf("part %d %s bytes=%zu\n", p, kBigNames[p], nbytes);
    }
    gguf_free(gguf);
    ggml_free(gctx);

    const int64_t N1 = 1408, NE = 64;
    const size_t row_bytes = (size_t)(K / 256) * 110; // 8*110 = 880
    const size_t expert_bytes = (size_t)N1 * row_bytes;
    const size_t stack_bytes = expert_bytes * (size_t)NE;
    if (parts[0].size() != stack_bytes) fail("gate nbytes mismatch");

    std::vector<uint8_t> wBig;
    wBig.reserve(stack_bytes * 4);
    for (int p = 0; p < 4; p++) {
        if (parts[p].size() != stack_bytes) fail("part size mismatch");
        wBig.insert(wBig.end(), parts[p].begin(), parts[p].end());
        if (p > 0) { std::vector<uint8_t>().swap(parts[p]); }
    }
    const uint8_t * wGate = parts[0].data();
    const int64_t nBig = N1 * NE * 4; // 360,448 rows

    // ---- deterministic activations, same sequence as q3k_ref.cpp ----
    std::mt19937 rng(1);
    std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
    std::vector<float> x_m1((size_t)K);
    for (auto & v : x_m1) v = uni(rng);
    std::vector<float> x_m8((size_t)K * 8);
    for (auto & v : x_m8) v = uni(rng); // column c at offset c*K (column-major)

    // ---- CPU reference via ggml dequant traits + f64 dots ----
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
    std::vector<float> y_e0_m1, y_e0_m8, y_st_m1, y_st_m8, y_bg_m1, y_bg_m8;
    cpu_gemv(wGate, N1, x_m1.data(), 1, y_e0_m1);
    cpu_gemv(wGate, N1, x_m8.data(), 8, y_e0_m8);
    cpu_gemv(wGate, N1 * NE, x_m1.data(), 1, y_st_m1);
    cpu_gemv(wGate, N1 * NE, x_m8.data(), 8, y_st_m8);
    cpu_gemv(wBig.data(), nBig, x_m1.data(), 1, y_bg_m1);
    cpu_gemv(wBig.data(), nBig, x_m8.data(), 8, y_bg_m8);

    // ---- write data files ----
    write_file(std::string(kDataDir) + "/gate.q3k", wGate, stack_bytes);
    write_file(std::string(kDataDir) + "/big.q3k", wBig.data(), wBig.size());
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
    write_file(std::string(kDataDir) + "/y_ref_big_m1.f32", y_bg_m1.data(),
               y_bg_m1.size() * sizeof(float));
    write_file(std::string(kDataDir) + "/y_ref_big_m8.f32", y_bg_m8.data(),
               y_bg_m8.size() * sizeof(float));
    std::printf("wrote data files to %s (big rows=%lld bytes=%zu)\n",
                kDataDir, (long long)nBig, wBig.size());

    if (data_only) return 0;

    // ---- CPU backend timing ----
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) fail("ggml_backend_cpu_init failed");

    struct Case {
        const char * shape;
        const uint8_t * w;
        int64_t nrows;
        const float * x;
        int m;
        const std::vector<float> & yref;
    };
    const Case cases[6] = {
        {"expert0", wGate, N1, x_m1.data(), 1, y_e0_m1},
        {"expert0", wGate, N1, x_m8.data(), 8, y_e0_m8},
        {"stack", wGate, N1 * NE, x_m1.data(), 1, y_st_m1},
        {"stack", wGate, N1 * NE, x_m8.data(), 8, y_st_m8},
        {"big", wBig.data(), nBig, x_m1.data(), 1, y_bg_m1},
        {"big", wBig.data(), nBig, x_m8.data(), 8, y_bg_m8},
    };
    const int kThreads[4] = {8, 16, 32, 64};
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

        const double denom = max_abs(cs.yref);
        std::vector<float> yg((size_t)cs.nrows * cs.m);
        for (int t : kThreads) {
            ggml_backend_cpu_set_n_threads(backend, t);
            for (int i = 0; i < 5; i++) ggml_backend_graph_compute(backend, gf);
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < 50; i++) ggml_backend_graph_compute(backend, gf);
            auto t1 = std::chrono::steady_clock::now();
            double us =
                std::chrono::duration<double, std::micro>(t1 - t0).count() / 50.0;

            // ggml y layout is [N x M]: element (r,c) at c*N + r
            ggml_backend_tensor_get(y, yg.data(), 0, yg.size() * sizeof(float));
            double maxerr = 0;
            for (int64_t r = 0; r < cs.nrows; r++)
                for (int c = 0; c < cs.m; c++) {
                    double d = std::fabs((double)yg[(size_t)c * cs.nrows + r] -
                                         cs.yref[(size_t)r * cs.m + c]);
                    maxerr = std::max(maxerr, d);
                }
            double gbs = (double)wn / (us * 1e-6) / 1e9;
            std::printf("CPUREF shape=%s m=%d threads=%d us=%.2f GB/s=%.2f max_rel_err=%.3e\n",
                        cs.shape, cs.m, t, us, gbs, maxerr / denom);
            std::fflush(stdout);
        }
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
    ggml_backend_free(backend);
    return 0;
}
