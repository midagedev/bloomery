// q3k_cpu_ref.cpp — CPU reference harness for mulle stage 2 pre-study (MUL-3).
//
// Stage 0's q3k_ref.cpp wrote gate.q3k (blk.1.ffn_gate_exps, Q3_K, 79.3 MB),
// the f32 activations x_m1/x_m8 and the CPU f32 references for expert0 and
// stack. This harness adds the `big` shape — blk.1 gate+up and blk.2 gate+up
// (4 x 79,298,560 B = 317,194,240 B, 360,448 rows, K=2048), extracted
// through the GGUF API — computes its f32 reference the same way q3k_ref did
// (ggml_internal_get_type_traits(GGML_TYPE_Q3_K)->to_float + double-precision
// f32 dots), and then times ggml's CPU backend over
// {expert0, stack, big} x M in {1, 8} x threads in {8, 16, 32, 64}:
// one ggml_mul_mat graph per call, ggml_backend_graph_compute, 5 warm-up +
// 50 timed iterations.
//
// Build: bash tools/ref/build-cpu.sh   (on the box, IK=/home/user/ik_llama.cpp)
// Run:   $MULLE_DATA/bin/q3k_cpu_ref   (normally via tools/ref/cpu-measure.sh,
//        under the machine-wide CPU lease)

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"

static const char * kGgufPath = "/models/small/DeepSeek-V2-Lite-Chat.Q3_K_M.gguf";
static const char * kDataDir = getenv("MULLE_DATA") ? getenv("MULLE_DATA") : "/root/mulle-data";

static const int64_t K = 2048;
static const int64_t N1 = 1408; // rows per expert
static const int64_t NE = 64;   // experts
static const size_t kRowBytes = 880;
static const size_t kExpertBytes = (size_t)N1 * kRowBytes;   // 1,239,040
static const size_t kStackBytes = kExpertBytes * (size_t)NE; // 79,298,560
static const size_t kBigBytes = 4 * kStackBytes;             // 317,194,240

static const char * kBigTensors[4] = {
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

static std::vector<uint8_t> read_exact(const std::string & path, size_t n, const char * what) {
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) fail(std::string("cannot open ") + what + ": " + path);
    std::vector<uint8_t> out(n);
    if (std::fread(out.data(), 1, n, f) != n) fail(std::string("short read ") + what + ": " + path);
    std::fclose(f);
    return out;
}

static std::vector<float> read_f32(const std::string & path, size_t nfloats, const char * what) {
    std::vector<uint8_t> b = read_exact(path, nfloats * sizeof(float), what);
    std::vector<float> out(nfloats);
    std::memcpy(out.data(), b.data(), b.size());
    return out;
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

static bool file_size_is(const std::string & path, size_t n) {
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    ::fseeko(f, 0, SEEK_END);
    bool ok = (size_t)::ftello(f) == n;
    std::fclose(f);
    return ok;
}

static std::string datapath(const char * name) {
    return std::string(kDataDir) + "/" + name;
}

int main() {
    // ---- 1. stage-0 data (produced by tools/ref/build.sh + q3k_ref) ----
    std::vector<uint8_t> gate = read_exact(datapath("gate.q3k").c_str(), kStackBytes, "gate.q3k");

    // Q3K_DBG_ROW=<n>: dequantize gate row n with ggml's to_float and dump the
    // f32 weights to dbg_row.f32 (port-debugging aid; exits without timing).
    if (getenv("Q3K_DBG_ROW")) {
        const int64_t row = atoll(getenv("Q3K_DBG_ROW"));
        ggml_type_traits_t tr = ggml_internal_get_type_traits(GGML_TYPE_Q3_K);
        std::vector<float> wrow((size_t)K);
        tr.to_float(gate.data() + (size_t)row * kRowBytes, wrow.data(), K);
        write_file(datapath("dbg_row.f32"), wrow.data(), wrow.size() * sizeof(float));
        std::printf("Q3K_DBG_ROW %lld dumped (%zu B)\n", (long long)row, wrow.size() * sizeof(float));
        return 0;
    }

    std::vector<float> x_m1 = read_f32(datapath("x_m1.f32").c_str(), (size_t)K, "x_m1.f32");
    std::vector<float> x_m8 = read_f32(datapath("x_m8.f32").c_str(), (size_t)K * 8, "x_m8.f32");
    std::vector<float> y_e0_m1 = read_f32(datapath("y_ref_expert0_m1.f32").c_str(), (size_t)N1, "y_ref_expert0_m1");
    std::vector<float> y_e0_m8 = read_f32(datapath("y_ref_expert0_m8.f32").c_str(), (size_t)N1 * 8, "y_ref_expert0_m8");
    std::vector<float> y_st_m1 = read_f32(datapath("y_ref_stack_m1.f32").c_str(), (size_t)N1 * NE, "y_ref_stack_m1");
    std::vector<float> y_st_m8 = read_f32(datapath("y_ref_stack_m8.f32").c_str(), (size_t)N1 * NE * 8, "y_ref_stack_m8");
    std::printf("loaded stage-0 data from %s\n", kDataDir);

    // ---- 2. big: extract the 4 tensors and compute its f32 reference ----
    // Cached: if big.q3k and both y_ref_big files exist with exact sizes, reuse.
    // (keep the strings alive: a .c_str() on a temporary dangles — first run
    // wrote the 317 MB extraction to a garbage path before this was caught)
    const std::string big_path = datapath("big.q3k");
    const std::string ybg_m1_path = datapath("y_ref_big_m1.f32");
    const std::string ybg_m8_path = datapath("y_ref_big_m8.f32");
    const int64_t big_rows = 4 * N1 * NE; // 360,448
    std::vector<uint8_t> big(kBigBytes);
    if (file_size_is(big_path, kBigBytes) && file_size_is(ybg_m1_path, (size_t)big_rows * 4) &&
        file_size_is(ybg_m8_path, (size_t)big_rows * 8 * 4)) {
        big = read_exact(big_path, kBigBytes, "big.q3k");
        std::printf("BIGREF reused %s (%zu B)\n", big_path.c_str(), kBigBytes);
    } else {
        // metadata-only gguf open (no_alloc), like q3k_ref
        struct ggml_context * gctx = nullptr;
        struct gguf_init_params gp = {true, &gctx};
        struct gguf_context * gguf = gguf_init_from_file(kGgufPath, gp);
        if (!gguf || !gctx) fail("gguf_init_from_file failed");
        const size_t data_off = gguf_get_data_offset(gguf);
        for (int t = 0; t < 4; ++t) {
            const char * name = kBigTensors[t];
            int tidx = gguf_find_tensor(gguf, name);
            if (tidx < 0) fail(std::string("tensor not found: ") + name);
            struct ggml_tensor * info = ggml_get_tensor(gctx, name);
            if (!info) fail(std::string("ggml_get_tensor failed: ") + name);
            if (info->type != GGML_TYPE_Q3_K) fail(std::string("not Q3_K: ") + name);
            if (info->ne[0] != K || info->ne[1] != N1 || info->ne[2] != NE)
                fail(std::string("unexpected dims: ") + name);
            if (ggml_nbytes(info) != kStackBytes) fail(std::string("nbytes mismatch: ") + name);
            std::vector<uint8_t> w =
                read_range(kGgufPath, (int64_t)(data_off + gguf_get_tensor_offset(gguf, tidx)), kStackBytes);
            std::memcpy(big.data() + (size_t)t * kStackBytes, w.data(), kStackBytes);
        }
        gguf_free(gguf);
        ggml_free(gctx);
        write_file(big_path, big.data(), kBigBytes);
        std::printf("BIGREF extracted %s (%zu B, %lld rows)\n", big_path.c_str(), kBigBytes, (long long)big_rows);

        // f32 reference: dequantize each row, double-precision dot per column
        // (same as q3k_ref's cpu_gemv).
        ggml_type_traits_t traits = ggml_internal_get_type_traits(GGML_TYPE_Q3_K);
        std::vector<float> wrow((size_t)K);
        auto cpu_gemv = [&](int m, std::vector<float> & y) {
            y.assign((size_t)big_rows * m, 0.0f);
            for (int64_t r = 0; r < big_rows; r++) {
                traits.to_float(big.data() + (size_t)r * kRowBytes, wrow.data(), K);
                const std::vector<float> & x = m == 1 ? x_m1 : x_m8;
                for (int c = 0; c < m; c++) {
                    const float * xc = x.data() + (size_t)c * (size_t)K;
                    double acc = 0;
                    for (int64_t k = 0; k < K; k++) acc += (double)wrow[(size_t)k] * xc[k];
                    y[(size_t)r * m + c] = (float)acc;
                }
            }
        };
        std::vector<float> y_m1, y_m8;
        cpu_gemv(1, y_m1);
        write_file(ybg_m1_path, y_m1.data(), y_m1.size() * sizeof(float));
        cpu_gemv(8, y_m8);
        write_file(ybg_m8_path, y_m8.data(), y_m8.size() * sizeof(float));
        std::printf("BIGREF wrote f32 references (%lld rows, M=1 and 8)\n", (long long)big_rows);
    }
    std::vector<float> y_bg_m1 = read_f32(ybg_m1_path, (size_t)big_rows, "y_ref_big_m1");
    std::vector<float> y_bg_m8 = read_f32(ybg_m8_path, (size_t)big_rows * 8, "y_ref_big_m8");

    // ---- 3. ggml CPU backend timing ----
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) fail("ggml_backend_cpu_init failed");

    auto max_abs = [](const std::vector<float> & v) {
        double m = 0;
        for (float f : v) m = std::max(m, (double)std::fabs(f));
        return m;
    };

    struct Case {
        const char * name;
        const uint8_t * w;
        int64_t nrows;
        const std::vector<float> * yref_m1;
        const std::vector<float> * yref_m8;
    };
    const Case cases[3] = {
        {"expert0", gate.data(), N1, &y_e0_m1, &y_e0_m8},
        {"stack", gate.data(), N1 * NE, &y_st_m1, &y_st_m8},
        {"big", big.data(), big_rows, &y_bg_m1, &y_bg_m8},
    };
    const int threads_list[4] = {8, 16, 32, 64};

    for (const Case & cs : cases) {
        for (int m : {1, 8}) {
            const std::vector<float> & yref = m == 1 ? *cs.yref_m1 : *cs.yref_m8;
            const std::vector<float> & x = m == 1 ? x_m1 : x_m8;
            // no_alloc: storage comes from ggml_backend_alloc_ctx_tensors
            struct ggml_init_params mp = {512u << 20, nullptr, true};
            struct ggml_context * ctx = ggml_init(mp);
            struct ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q3_K, K, cs.nrows);
            struct ggml_tensor * xt = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, m);
            struct ggml_tensor * y = ggml_mul_mat(ctx, w, xt);
            struct ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, y);
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
            if (!buf) fail("alloc_ctx_tensors failed");
            const size_t wn = (size_t)cs.nrows * kRowBytes;
            ggml_backend_tensor_set(w, cs.w, 0, wn);
            ggml_backend_tensor_set(xt, x.data(), 0, (size_t)K * m * sizeof(float));

            for (int t : threads_list) {
                ggml_backend_cpu_set_n_threads(backend, t);
                for (int i = 0; i < 5; i++) ggml_backend_graph_compute(backend, gf);
                ggml_backend_synchronize(backend);
                auto t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < 50; i++) ggml_backend_graph_compute(backend, gf);
                ggml_backend_synchronize(backend);
                auto t1 = std::chrono::steady_clock::now();
                double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / 50.0;

                // ggml y layout is [N x M]: element (r,c) at c*N + r
                std::vector<float> yg((size_t)cs.nrows * m);
                ggml_backend_tensor_get(y, yg.data(), 0, yg.size() * sizeof(float));
                double denom = max_abs(yref);
                double maxerr = 0;
                for (int64_t r = 0; r < cs.nrows; r++)
                    for (int c = 0; c < m; c++) {
                        double d = std::fabs((double)yg[(size_t)c * cs.nrows + r] -
                                             yref[(size_t)r * m + c]);
                        maxerr = std::max(maxerr, d);
                    }
                double gbs = (double)wn / (us * 1e-6) / 1e9;
                std::printf("CPUREF shape=%s m=%d threads=%d us=%.2f GB/s=%.2f max_rel_err=%.3e\n",
                            cs.name, m, t, us, gbs, maxerr / denom);
            }
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
        }
    }
    ggml_backend_free(backend);
    return 0;
}
