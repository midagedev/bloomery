// mulle stage 0 — Q3_K gate-tensor reference harness (runs on the box).
//
// Extracts blk.1.ffn_gate_exps.weight (Q3_K, [2048 x 1408 x 64]) from the
// DeepSeek-V2-Lite GGUF through the gguf API, writes the raw tensor plus
// deterministic f32 activations to /root/mulle-data/, computes the CPU f32
// reference outputs with ggml's own dequantize (type_traits.to_float) and
// plain f32 dot products, and then times ggml's CUDA mul_mat (the mmvq path
// taken for M<=8) for the four stage-0 shapes:
//
//   expert0 : one expert  [2048 x 1408]
//   stack   : all experts [2048 x 90112]
//   M = 1 and M = 8 activation columns
//
// Per shape it prints µs/launch, GB/s (weight bytes / time) and the max
// relative error against the CPU reference. ggml's error is ~1e-2 by
// construction (it quantizes the activation to q8_1); that is recorded, not
// treated as failure.
//
// build: tools/ref/build.sh    run: /root/mulle-data/q3k_ref

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <vector>

namespace {

constexpr int64_t K       = 2048;               // k dimension of the tensor
constexpr int64_t N_EXP   = 1408;               // rows per expert
constexpr int64_t NEXP    = 64;                 // experts
constexpr int64_t N_STACK = N_EXP * NEXP;       // 90112 rows
constexpr size_t  ROW_B   = 110 * (K / 256);    // 880 bytes per row

const char * kTensorName = "blk.1.ffn_gate_exps.weight";
const char * kDataDir    = "/root/mulle-data";

// ---------------------------------------------------------------------------
// small utilities

[[noreturn]] void die(const std::string & msg) {
    fprintf(stderr, "q3k_ref: %s\n", msg.c_str());
    exit(1);
}

// deterministic uniform [-1, 1) floats from an explicit LCG (seed 1), so the
// exact activation bytes are reproducible from this file alone
struct Lcg {
    uint64_t s = 1;
    float next() {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        double u = (double)(s >> 11) * (1.0 / 9007199254740992.0); // 53 bits
        return (float)(2.0 * u - 1.0);
    }
};

void write_file(const std::string & path, const void * data, size_t nbytes) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) die("cannot open " + path + " for writing");
    if (fwrite(data, 1, nbytes, f) != nbytes) die("short write to " + path);
    fclose(f);
    printf("wrote %s (%zu bytes)\n", path.c_str(), nbytes);
}

void run_cmd(const char * cmd) {
    FILE * p = popen(cmd, "r");
    if (!p) return;
    std::string out;
    char buf[512];
    while (fgets(buf, sizeof buf, p)) out += buf;
    pclose(p);
    fputs(out.c_str(), stdout);
}

// quiet-machine witness: both cards, load average, io pressure (avg10)
void witnesses(const char * when) {
    printf("[witness %s]\n", when);
    run_cmd("nvidia-smi --query-gpu=name,memory.used,utilization.gpu,power.draw --format=csv");
    run_cmd("cat /proc/loadavg");
    run_cmd("grep -E '^(some|full)' /proc/pressure/io");
    fflush(stdout);
}

// ---------------------------------------------------------------------------
// CPU reference: y[n][m] = sum_k deq(w)[n][k] * x[m][k], plain f32

// rows are split across threads; each output element is computed exactly as
// in the single-threaded form (dequantize row, sequential f32 dot)
void ref_dots(const uint8_t * w, int64_t n_rows, const float * x, int m,
              float * y /* [n_rows][m] */, const ggml_to_float_t to_float,
              int64_t row_begin, int64_t row_end) {
    std::vector<float> row(K);
    for (int64_t n = row_begin; n < row_end; ++n) {
        to_float(w + n * ROW_B, row.data(), K);
        for (int mi = 0; mi < m; ++mi) {
            const float * xr = x + (size_t)mi * K;
            float acc = 0.0f;
            for (int64_t k = 0; k < K; ++k) acc += row[k] * xr[k];
            y[n * m + mi] = acc;
        }
    }
}

std::vector<float> cpu_reference(const uint8_t * w, int64_t n_rows,
                                 const float * x, int m,
                                 const ggml_to_float_t to_float) {
    std::vector<float> y((size_t)n_rows * m);
    unsigned nthreads = std::min<unsigned>(8, std::thread::hardware_concurrency());
    std::vector<std::thread> ts;
    int64_t chunk = (n_rows + nthreads - 1) / nthreads;
    for (unsigned t = 0; t < nthreads; ++t) {
        int64_t b = (int64_t)t * chunk, e = std::min<int64_t>(n_rows, b + chunk);
        if (b >= e) break;
        ts.emplace_back(ref_dots, w, n_rows, x, m, y.data(), to_float, b, e);
    }
    for (auto & th : ts) th.join();
    return y;
}

// ---------------------------------------------------------------------------
// ggml CUDA mul_mat timing for one shape

struct Timing { double us_per_launch; double gbps; double max_rel_err; };

Timing time_ggml(const uint8_t * w_bytes, size_t w_nbytes, int64_t n_rows,
                 const float * x, int m, const std::vector<float> & y_ref) {
    const size_t mem_size = ggml_tensor_overhead() * 8 + ggml_graph_overhead() + 1024 * 1024;
    std::vector<uint8_t> mem(mem_size);
    struct ggml_init_params ip = { mem_size, mem.data(), /*no_alloc*/ true };
    struct ggml_context * ctx = ggml_init(ip);
    if (!ctx) die("ggml_init failed");

    const int64_t ne_a[2] = { K, n_rows };
    const int64_t ne_b[2] = { K, m };
    struct ggml_tensor * a = ggml_new_tensor(ctx, GGML_TYPE_Q3_K, 2, ne_a);
    struct ggml_tensor * b = ggml_new_tensor(ctx, GGML_TYPE_F32,   2, ne_b);
    struct ggml_tensor * d = ggml_mul_mat(ctx, a, b);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, d);

    ggml_backend_t backend = ggml_backend_cuda_init(0, nullptr, nullptr);
    if (!backend) die("ggml_backend_cuda_init failed");
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(galloc, gf)) die("gallocr alloc failed");

    ggml_backend_tensor_set(a, w_bytes, 0, w_nbytes);
    ggml_backend_tensor_set(b, x, 0, (size_t)K * m * sizeof(float));

    // warm-up
    for (int i = 0; i < 20; ++i) {
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS)
            die("ggml_backend_graph_compute failed (warm-up)");
    }
    ggml_backend_synchronize(backend);

    // 200 launches back-to-back, one synchronize at the end. ggml computes
    // on its own non-blocking stream, so CUDA events on the default stream
    // do not order against it; wall clock across the batch + full device
    // synchronize is the honest equivalent (first measured 2026-09-19:
    // events gave 0.000 us and a zeroed readback).
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 200; ++i) {
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS)
            die("ggml_backend_graph_compute failed");
    }
    ggml_backend_synchronize(backend);
    auto t1 = std::chrono::steady_clock::now();

    std::vector<float> y((size_t)n_rows * m);
    // ggml mul_mat result is [m, n_rows]; our reference is [n_rows][m]
    ggml_backend_tensor_get(d, y.data(), 0, y.size() * sizeof(float));

    double max_abs_ref = 0.0, max_abs_err = 0.0;
    for (int64_t n = 0; n < n_rows; ++n) {
        for (int mi = 0; mi < m; ++mi) {
            double ref = y_ref[n * m + mi];
            double got = y[(size_t)mi * n_rows + n];
            max_abs_ref = std::max(max_abs_ref, std::fabs(ref));
            max_abs_err = std::max(max_abs_err, std::fabs(got - ref));
        }
    }
    // first-eight sample for eyeballing layout/garbage problems
    printf("  sample n=0..3 m=0: gpu");
    for (int n = 0; n < 4; ++n) printf(" %.6e", y[n]);
    printf("\n  sample n=0..3 m=0: ref");
    for (int n = 0; n < 4; ++n) printf(" %.6e", y_ref[n * m]);
    printf("\n");

    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / 200.0;
    Timing t;
    t.us_per_launch = us;
    t.gbps = (double)w_nbytes / (us * 1e-6) / 1e9;
    t.max_rel_err = max_abs_err / max_abs_ref;

    ggml_gallocr_free(galloc);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return t;
}

} // namespace

int main() {
    const char * model = std::getenv("MODEL");
    if (!model) model = "/models/small/DeepSeek-V2-Lite-Chat.Q3_K_M.gguf";

    // ggml's fp16->f32 lookup table (used by Q3_K dequantization) is only
    // populated by ggml_init(); without this the CPU reference comes out as
    // signed zeros (first measured 2026-09-19: probe dequant sum=0.0).
    {
        const size_t probe_mem = ggml_tensor_overhead() * 2 + 1024 * 1024;
        std::vector<uint8_t> mem(probe_mem);
        struct ggml_init_params ip = { probe_mem, mem.data(), /*no_alloc*/ true };
        if (!ggml_init(ip)) die("ggml_init failed");
    }

    mkdir(kDataDir, 0755); // ok if it exists

    // ---- extract the tensor through the gguf API --------------------------
    struct gguf_init_params gp = { /*no_alloc*/ true, /*ctx*/ nullptr };
    struct gguf_context * gctx = gguf_init_from_file(model, gp);
    if (!gctx) die(std::string("gguf_init_from_file failed for ") + model);

    int ti = gguf_find_tensor(gctx, kTensorName);
    if (ti < 0) die(std::string("tensor not found: ") + kTensorName);
    if (gguf_get_tensor_type(gctx, ti) != GGML_TYPE_Q3_K) die("tensor is not Q3_K");

    const size_t nbytes = (size_t)N_STACK * ROW_B; // 79,298,560
    const size_t data_off = gguf_get_data_offset(gctx);
    const size_t tensor_off = gguf_get_tensor_offset(gctx, ti);

    FILE * f = fopen(model, "rb");
    if (!f) die("cannot reopen model file");
    fseeko(f, (off_t)(data_off + tensor_off), SEEK_SET);
    std::vector<uint8_t> w(nbytes);
    if (fread(w.data(), 1, nbytes, f) != nbytes) die("short read on tensor data");
    fclose(f);
    gguf_free(gctx);

    printf("extracted %s: %zu bytes (%lld rows x %zu bytes)\n", kTensorName,
           nbytes, (long long)N_STACK, ROW_B);
    write_file(std::string(kDataDir) + "/gate.q3k", w.data(), nbytes);

    // ---- deterministic activations ----------------------------------------
    Lcg lcg;
    std::vector<float> x1(K);
    for (auto & v : x1) v = lcg.next();
    std::vector<float> x8((size_t)K * 8);
    for (auto & v : x8) v = lcg.next();
    write_file(std::string(kDataDir) + "/x_m1.f32", x1.data(), x1.size() * 4);
    write_file(std::string(kDataDir) + "/x_m8.f32", x8.data(), x8.size() * 4);

    // ---- CPU reference (the truth both GPU paths are compared to) ---------
    const ggml_type_traits_t tt = ggml_internal_get_type_traits(GGML_TYPE_Q3_K);
    if (!tt.to_float) die("Q3_K to_float unavailable");
    {
        std::vector<float> probe(K);
        tt.to_float(w.data(), probe.data(), K);
        double sum = 0.0;
        for (auto v : probe) sum += v;
        printf("probe dequant row0[0..3]: %.6e %.6e %.6e %.6e sum=%.6e | x1[0..3]: %.6f %.6f %.6f %.6f\n",
               probe[0], probe[1], probe[2], probe[3], sum, x1[0], x1[1], x1[2], x1[3]);
    }
    printf("computing CPU references with ggml to_float=%s dequantization...\n",
           ggml_type_name(GGML_TYPE_Q3_K));

    struct Shape { const char * name; const uint8_t * w; size_t nbytes; int64_t rows; };
    const Shape shapes[2] = {
        { "expert0", w.data(), (size_t)N_EXP * ROW_B, N_EXP },
        { "stack",   w.data(), nbytes,                 N_STACK },
    };

    std::vector<float> y_ref[2][2]; // [shape][m_idx]
    for (int s = 0; s < 2; ++s) {
        y_ref[s][0] = cpu_reference(shapes[s].w, shapes[s].rows, x1.data(), 1, tt.to_float);
        y_ref[s][1] = cpu_reference(shapes[s].w, shapes[s].rows, x8.data(), 8, tt.to_float);
        std::string base = std::string(kDataDir) + "/y_ref_" + shapes[s].name;
        write_file(base + "_m1.f32", y_ref[s][0].data(), y_ref[s][0].size() * 4);
        write_file(base + "_m8.f32", y_ref[s][1].data(), y_ref[s][1].size() * 4);
    }

    // ---- ggml CUDA timing, all four shapes back-to-back --------------------
    witnesses("before ggml timing");
    for (int s = 0; s < 2; ++s) {
        for (int mi = 0; mi < 2; ++mi) {
            int m = mi == 0 ? 1 : 8;
            Timing t = time_ggml(shapes[s].w, shapes[s].nbytes, shapes[s].rows,
                                 mi == 0 ? x1.data() : x8.data(), m, y_ref[s][mi]);
            printf("RESULT engine=ggml shape=%s m=%d us_per_launch=%.3f gbps=%.1f max_rel_err=%.3e\n",
                   shapes[s].name, m, t.us_per_launch, t.gbps, t.max_rel_err);
        }
    }
    witnesses("after ggml timing");

    printf("done\n");
    return 0;
}
