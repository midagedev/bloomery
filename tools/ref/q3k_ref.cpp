// q3k_ref.cpp — reference harness for mulle stage 0 (MUL-9 adds Q4_K/Q6_K).
//
// Opens the DeepSeek-V2-Lite-Chat Q3_K_M GGUF, extracts the tensors below,
// writes raw bytes + f32 activations to $MULLE_DATA/ (default
// /root/mulle-data), computes CPU reference outputs by dequantizing with
// ggml_internal_get_type_traits(<type>)->to_float and plain f32 dots, then
// times ggml's CUDA mul_mat on every shape and compares.
//
// Shapes (K=2048 fixed), M = 1 and 8 activation columns each:
//   Q3_K expert0/stack : blk.1.ffn_gate_exps.weight, N=1408 / 90112
//   Q4_K attn0/attnstk : the 27 blk.*.attn_output [2048,2048] tensors,
//                        concatenated in blk order; attn0 = the first
//                        tensor alone (N=2048), attnstk = all 27 (N=55296)
//   Q6_K head          : output.weight [2048,102400], N=102400
//
// MUL-8 amortization family (2026-09-19): additionally stack_k{1,2,3,4,6,8},
// attnstk_k{...}, head_k{...} — the FIRST k columns of the SAME x_m8 draw, so
// the family is nested (M=1 ⊂ M=2 ⊂ ... ⊂ M=8) and the curve is a statement
// about M alone. The existing rows and their names are untouched; k1 is a new
// row (x_m8's first column), NOT the existing *_m1 (a different draw).
//
// The timing loop also dumps every case's µs to $MULLE_DATA/ggml_timings.txt
// (truncated each run) so the rust binary can anchor its ratio gate to ggml's
// own ratio from the SAME measure.sh invocation.
//
// The 26 ffn_down_shexp tensors are also Q4_K but K=2816; out of scope here
// (the kernels are K=2048).
//
// Build: bash tools/ref/build.sh   (on the box, IK=/home/user/ik_llama.cpp)
// Run:   LD_LIBRARY_PATH=$IK/build/ggml/src ./tools/ref/q3k_ref

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
#include "ggml-cuda.h"

static const char * kTensorName = "blk.1.ffn_gate_exps.weight";
static const char * kGgufPath = "/models/small/DeepSeek-V2-Lite-Chat.Q3_K_M.gguf";
static const char * kDataDir = getenv("MULLE_DATA") ? getenv("MULLE_DATA") : "/root/mulle-data";

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

    // ---- Q4_K: the 27 blk.*.attn_output [2048,2048] tensors, concatenated
    // in blk order (the tensors are not adjacent in the file, so read each
    // through its own offset). attn0 = tensor 0 alone, attnstk = all 27. ----
    const int64_t N_ATTN = 2048;
    const int64_t N_STK = 27 * N_ATTN; // 55296
    const size_t rb4 = (size_t)(K / 256) * 144; // 8*144 = 1152
    std::vector<uint8_t> wAttn((size_t)N_STK * rb4);
    for (int b = 0; b < 27; b++) {
        char name[64];
        std::snprintf(name, sizeof name, "blk.%d.attn_output.weight", b);
        int ti = gguf_find_tensor(gguf, name);
        if (ti < 0) fail(std::string("tensor not found: ") + name);
        struct ggml_tensor * t = ggml_get_tensor(gctx, name);
        if (!t) fail(std::string("ggml_get_tensor failed: ") + name);
        if (t->type != GGML_TYPE_Q4_K)
            fail(std::string(name) + " is not Q4_K");
        if (t->ne[0] != K || t->ne[1] != N_ATTN)
            fail(std::string(name) + " unexpected dims");
        std::vector<uint8_t> one = read_range(
            kGgufPath, (int64_t)(data_off + gguf_get_tensor_offset(gguf, ti)),
            (size_t)N_ATTN * rb4);
        std::memcpy(&wAttn[(size_t)b * (size_t)N_ATTN * rb4], one.data(), one.size());
    }

    // ---- Q6_K: output.weight [2048,102400] ----
    static const char * kOutName = "output.weight";
    int oidx = gguf_find_tensor(gguf, kOutName);
    if (oidx < 0) fail("output.weight not found");
    struct ggml_tensor * ot = ggml_get_tensor(gctx, kOutName);
    if (!ot) fail("ggml_get_tensor failed for output.weight");
    if (ot->type != GGML_TYPE_Q6_K) fail("output.weight is not Q6_K");
    const int64_t N_HEAD = ot->ne[1];
    if (ot->ne[0] != K || N_HEAD != 102400)
        fail("output.weight unexpected dims");
    const size_t rb6 = (size_t)(K / 256) * 210; // 8*210 = 1680
    std::vector<uint8_t> wHead = read_range(
        kGgufPath, (int64_t)(data_off + gguf_get_tensor_offset(gguf, oidx)),
        (size_t)N_HEAD * rb6);

    gguf_free(gguf);
    ggml_free(gctx);

    // ---- deterministic activations: mt19937 seed 1, uniform [-1,1] ----
    std::mt19937 rng(1);
    std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
    std::vector<float> x_m1((size_t)K);
    for (auto & v : x_m1) v = uni(rng);
    std::vector<float> x_m8((size_t)K * 8);
    for (auto & v : x_m8) v = uni(rng); // column c at offset c*K (column-major)

    // ---- CPU reference via ggml dequant traits + f32 dots (double
    // accumulators); one body parameterized by type so Q3_K/Q4_K/Q6_K share
    // it. Activation order is untouched: x_m1 then x_m8, seed 1. ----
    const ggml_type_traits_t tr3 = ggml_internal_get_type_traits(GGML_TYPE_Q3_K);
    const ggml_type_traits_t tr4 = ggml_internal_get_type_traits(GGML_TYPE_Q4_K);
    const ggml_type_traits_t tr6 = ggml_internal_get_type_traits(GGML_TYPE_Q6_K);
    std::vector<float> wrow((size_t)K);
    auto cpu_gemv = [&](ggml_type_traits_t tr, size_t rbytes, const uint8_t * w,
                        int64_t nrows, const float * x,
                        int m, std::vector<float> & y /* [nrows*m] row-major */) {
        y.assign((size_t)nrows * m, 0.0f);
        for (int64_t r = 0; r < nrows; r++) {
            tr.to_float(w + (size_t)r * rbytes, wrow.data(), K);
            for (int c = 0; c < m; c++) {
                const float * xc = x + (size_t)c * (size_t)K;
                double acc = 0.0;
                for (int64_t k = 0; k < K; k++) acc += (double)wrow[(size_t)k] * xc[k];
                y[(size_t)r * m + c] = (float)acc;
            }
        }
    };
    std::vector<float> y_e0_m1, y_e0_m8, y_st_m1, y_st_m8;
    cpu_gemv(tr3, row_bytes, wAll.data(), N1, x_m1.data(), 1, y_e0_m1);
    cpu_gemv(tr3, row_bytes, wAll.data(), N1, x_m8.data(), 8, y_e0_m8);
    cpu_gemv(tr3, row_bytes, wAll.data(), N1 * NE, x_m1.data(), 1, y_st_m1);
    cpu_gemv(tr3, row_bytes, wAll.data(), N1 * NE, x_m8.data(), 8, y_st_m8);
    std::vector<float> y_at0_m1, y_at0_m8, y_ast_m1, y_ast_m8, y_h_m1, y_h_m8;
    cpu_gemv(tr4, rb4, wAttn.data(), N_ATTN, x_m1.data(), 1, y_at0_m1);
    cpu_gemv(tr4, rb4, wAttn.data(), N_ATTN, x_m8.data(), 8, y_at0_m8);
    cpu_gemv(tr4, rb4, wAttn.data(), N_STK, x_m1.data(), 1, y_ast_m1);
    cpu_gemv(tr4, rb4, wAttn.data(), N_STK, x_m8.data(), 8, y_ast_m8);
    cpu_gemv(tr6, rb6, wHead.data(), N_HEAD, x_m1.data(), 1, y_h_m1);
    cpu_gemv(tr6, rb6, wHead.data(), N_HEAD, x_m8.data(), 8, y_h_m8);

    // ---- MUL-8 amortization family: k columns of x_m8 (nested prefix), the
    // same cpu_gemv body. k8 recomputes the m8 dots under its own name so the
    // family is complete; the y_ref_*_m8 files above stay the gated rows. ----
    static const int kKs[6] = {1, 2, 3, 4, 6, 8};
    std::vector<float> y_st_k[6], y_ast_k[6], y_h_k[6];
    for (int i = 0; i < 6; i++) {
        cpu_gemv(tr3, row_bytes, wAll.data(), N1 * NE, x_m8.data(), kKs[i], y_st_k[i]);
        cpu_gemv(tr4, rb4, wAttn.data(), N_STK, x_m8.data(), kKs[i], y_ast_k[i]);
        cpu_gemv(tr6, rb6, wHead.data(), N_HEAD, x_m8.data(), kKs[i], y_h_k[i]);
    }

    // ---- write data files ----
    write_file(std::string(kDataDir) + "/gate.q3k", wAll.data(), total_bytes);
    write_file(std::string(kDataDir) + "/attn.q4k", wAttn.data(), wAttn.size());
    write_file(std::string(kDataDir) + "/output.q6k", wHead.data(), wHead.size());
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
    write_file(std::string(kDataDir) + "/y_ref_attn0_m1.f32", y_at0_m1.data(),
               y_at0_m1.size() * sizeof(float));
    write_file(std::string(kDataDir) + "/y_ref_attn0_m8.f32", y_at0_m8.data(),
               y_at0_m8.size() * sizeof(float));
    write_file(std::string(kDataDir) + "/y_ref_attnstk_m1.f32", y_ast_m1.data(),
               y_ast_m1.size() * sizeof(float));
    write_file(std::string(kDataDir) + "/y_ref_attnstk_m8.f32", y_ast_m8.data(),
               y_ast_m8.size() * sizeof(float));
    write_file(std::string(kDataDir) + "/y_ref_head_m1.f32", y_h_m1.data(),
               y_h_m1.size() * sizeof(float));
    write_file(std::string(kDataDir) + "/y_ref_head_m8.f32", y_h_m8.data(),
               y_h_m8.size() * sizeof(float));
    for (int i = 0; i < 6; i++) {
        char nm[64];
        std::snprintf(nm, sizeof nm, "y_ref_stack_k%d.f32", kKs[i]);
        write_file(std::string(kDataDir) + "/" + nm, y_st_k[i].data(),
                   y_st_k[i].size() * sizeof(float));
        std::snprintf(nm, sizeof nm, "y_ref_attnstk_k%d.f32", kKs[i]);
        write_file(std::string(kDataDir) + "/" + nm, y_ast_k[i].data(),
                   y_ast_k[i].size() * sizeof(float));
        std::snprintf(nm, sizeof nm, "y_ref_head_k%d.f32", kKs[i]);
        write_file(std::string(kDataDir) + "/" + nm, y_h_k[i].data(),
                   y_h_k[i].size() * sizeof(float));
    }
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
        ggml_type ty;
        size_t rbytes;
        const uint8_t * w;
        int64_t nrows;
        const float * x;
        int m;
        const std::vector<float> & yref;
    };
    const Case cases[28] = {
        {"expert0_m1", GGML_TYPE_Q3_K, row_bytes, wAll.data(), N1, x_m1.data(), 1, y_e0_m1},
        {"expert0_m8", GGML_TYPE_Q3_K, row_bytes, wAll.data(), N1, x_m8.data(), 8, y_e0_m8},
        {"stack_m1", GGML_TYPE_Q3_K, row_bytes, wAll.data(), N1 * NE, x_m1.data(), 1, y_st_m1},
        {"stack_m8", GGML_TYPE_Q3_K, row_bytes, wAll.data(), N1 * NE, x_m8.data(), 8, y_st_m8},
        {"attn0_m1", GGML_TYPE_Q4_K, rb4, wAttn.data(), N_ATTN, x_m1.data(), 1, y_at0_m1},
        {"attn0_m8", GGML_TYPE_Q4_K, rb4, wAttn.data(), N_ATTN, x_m8.data(), 8, y_at0_m8},
        {"attnstk_m1", GGML_TYPE_Q4_K, rb4, wAttn.data(), N_STK, x_m1.data(), 1, y_ast_m1},
        {"attnstk_m8", GGML_TYPE_Q4_K, rb4, wAttn.data(), N_STK, x_m8.data(), 8, y_ast_m8},
        {"head_m1", GGML_TYPE_Q6_K, rb6, wHead.data(), N_HEAD, x_m1.data(), 1, y_h_m1},
        {"head_m8", GGML_TYPE_Q6_K, rb6, wHead.data(), N_HEAD, x_m8.data(), 8, y_h_m8},
        // Amortization family: first k columns of the same x_m8 draw.
        {"stack_k1", GGML_TYPE_Q3_K, row_bytes, wAll.data(), N1 * NE, x_m8.data(), 1, y_st_k[0]},
        {"stack_k2", GGML_TYPE_Q3_K, row_bytes, wAll.data(), N1 * NE, x_m8.data(), 2, y_st_k[1]},
        {"stack_k3", GGML_TYPE_Q3_K, row_bytes, wAll.data(), N1 * NE, x_m8.data(), 3, y_st_k[2]},
        {"stack_k4", GGML_TYPE_Q3_K, row_bytes, wAll.data(), N1 * NE, x_m8.data(), 4, y_st_k[3]},
        {"stack_k6", GGML_TYPE_Q3_K, row_bytes, wAll.data(), N1 * NE, x_m8.data(), 6, y_st_k[4]},
        {"stack_k8", GGML_TYPE_Q3_K, row_bytes, wAll.data(), N1 * NE, x_m8.data(), 8, y_st_k[5]},
        {"attnstk_k1", GGML_TYPE_Q4_K, rb4, wAttn.data(), N_STK, x_m8.data(), 1, y_ast_k[0]},
        {"attnstk_k2", GGML_TYPE_Q4_K, rb4, wAttn.data(), N_STK, x_m8.data(), 2, y_ast_k[1]},
        {"attnstk_k3", GGML_TYPE_Q4_K, rb4, wAttn.data(), N_STK, x_m8.data(), 3, y_ast_k[2]},
        {"attnstk_k4", GGML_TYPE_Q4_K, rb4, wAttn.data(), N_STK, x_m8.data(), 4, y_ast_k[3]},
        {"attnstk_k6", GGML_TYPE_Q4_K, rb4, wAttn.data(), N_STK, x_m8.data(), 6, y_ast_k[4]},
        {"attnstk_k8", GGML_TYPE_Q4_K, rb4, wAttn.data(), N_STK, x_m8.data(), 8, y_ast_k[5]},
        {"head_k1", GGML_TYPE_Q6_K, rb6, wHead.data(), N_HEAD, x_m8.data(), 1, y_h_k[0]},
        {"head_k2", GGML_TYPE_Q6_K, rb6, wHead.data(), N_HEAD, x_m8.data(), 2, y_h_k[1]},
        {"head_k3", GGML_TYPE_Q6_K, rb6, wHead.data(), N_HEAD, x_m8.data(), 3, y_h_k[2]},
        {"head_k4", GGML_TYPE_Q6_K, rb6, wHead.data(), N_HEAD, x_m8.data(), 4, y_h_k[3]},
        {"head_k6", GGML_TYPE_Q6_K, rb6, wHead.data(), N_HEAD, x_m8.data(), 6, y_h_k[4]},
        {"head_k8", GGML_TYPE_Q6_K, rb6, wHead.data(), N_HEAD, x_m8.data(), 8, y_h_k[5]},
    };
    // Per-case µs dump for the rust binary's same-invocation ratio gate
    // (truncated every run, so a stale file can never pass for fresh numbers).
    FILE * tdump = std::fopen((std::string(kDataDir) + "/ggml_timings.txt").c_str(), "w");
    if (!tdump) fail("cannot open ggml_timings.txt for write");
    for (const Case & cs : cases) {
        // no_alloc: tensor storage comes from ggml_backend_alloc_ctx_tensors
        struct ggml_init_params mp = {256u << 20, nullptr, true};
        struct ggml_context * ctx = ggml_init(mp);
        struct ggml_tensor * w =
            ggml_new_tensor_2d(ctx, cs.ty, K, cs.nrows);
        struct ggml_tensor * x =
            ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, cs.m);
        struct ggml_tensor * y = ggml_mul_mat(ctx, w, x);
        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, y);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buf) fail("alloc_ctx_tensors failed");
        const size_t wn = (size_t)cs.nrows * cs.rbytes;
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
        std::fprintf(tdump, "%s %.3f\n", cs.name, us);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
    std::fclose(tdump);
    ggml_backend_free(backend);
    return 0;
}
